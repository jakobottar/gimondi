import torch
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
import pynvml
from torchvision.transforms import ColorJitter
import utils
import namegenerator


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        validation_dataloader,
        loss,
        optimizer,
        n_gpu=2,
        mode="semi",
        name="None",
    ) -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.sup_loss_fn = loss
        self.unsup_loss_fn = torch.nn.MSELoss()

        if name:
            self.name = name
        else:
            self.name = namegenerator.gen()

        self.writer = SummaryWriter("runs/" + self.name)
        print(f"run name: {self.name}")
        os.makedirs(f"./out/{self.name}/", exist_ok=True)

        self.curr_epoch = 0
        self.unsup_weight = lambda e: min(e * 1e-3, 0.03)

        # set device
        # TODO: migrate to DistributedDataParallel
        self.device, gpu_ids = self._get_available_devices(n_gpu)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)

    def train(self, epochs: int):
        def schedule(e):
            if e < 5:
                return "super"
            if e % 2 == 1:
                return "semi"
            if e % 2 == 1:
                return "semi"
            return "super"

        print(f"training for {epochs} epochs: ")
        for e in range(epochs):
            self.curr_epoch = e
            self._train_epoch(mode=schedule(e))
            # self._train_epoch(mode = "super")
            (images, masks), _ = next(iter(self.train_dataloader))
            self._example_image(images, masks, prefix="train")

            self._valid_epoch()
            images, masks = next(iter(self.valid_dataloader))
            self._example_image(images, masks, prefix="valid")

            torch.save(
                self.model.module.state_dict(), f"./out/{self.name}/chkpt_{e}.pth"
            )
        torch.save(self.model.module.state_dict(), f"./out/{self.name}/model.pth")
        self.writer.close()

    def _train_epoch(self, mode):
        print(f"training mode: {mode}")
        self.model.train()
        # we need to do this each time to "reset" the iter obj.
        dataloader = iter(self.train_dataloader)

        term_size = shutil.get_terminal_size()
        tbar = tqdm(range(len(self.train_dataloader)), ncols=term_size.columns)

        for batch_idx in tbar:
            (image_l, mask), image_ul = next(dataloader)
            image_l = image_l.cuda(device=self.device, non_blocking=True)
            mask = mask.cuda(device=self.device, non_blocking=True)

            # Compute prediction error
            sup_preds = self.model(image_l)
            loss = self.sup_loss_fn(sup_preds, mask.squeeze(1).long())
            s_loss = loss.item()

            # unsupervised section
            if mode == "semi":
                image_ul = image_ul.cuda(device=self.device, non_blocking=True)

                target = image_ul.clone().detach()
                target = self.model(target)
                target = F.softmax(target, dim=1)

                # fig = plt.figure(figsize=(6, 6))
                # ax1 = plt.subplot(2, 2, 1)
                # ax2 = plt.subplot(2, 2, 2)

                # ax3 = plt.subplot(2, 2, 3)
                # ax4 = plt.subplot(2, 2, 4, sharex=ax2, sharey=ax2)

                # ax1.imshow(image_ul[0].cpu().numpy().transpose(1, 2, 0))
                # ax1.set_axis_off()
                # ax1.set_title("Original Image")

                # ax2.imshow(target[0][0].cpu().detach().numpy())
                # ax2.set_axis_off()
                # ax2.set_title("Target Mask")

                jitter = ColorJitter(brightness=0.25, contrast=0.25)
                image_ul = jitter(image_ul)

                # ax3.imshow(image_ul[0].cpu().numpy().transpose(1, 2, 0))
                # ax3.set_axis_off()
                # ax3.set_title("Jittered Image")

                pred_ul = self.model(image_ul)
                us_loss = self.unsup_loss_fn(target, pred_ul)

                # pred_ul = F.softmax(pred_ul, dim=1)
                # ax4.imshow(pred_ul[0][0].cpu().detach().numpy())
                # ax4.set_axis_off()
                # ax4.set_title("Unsupervised Prediction")
                # plt.savefig("./out/test.png")

                loss += self.unsup_weight(self.curr_epoch) * us_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                itr = (self.curr_epoch * len(dataloader)) + batch_idx

                self.writer.add_scalar("supervised_loss", s_loss, itr)

                if mode == "semi":
                    self.writer.add_scalar("unsupervised_loss", us_loss, itr)
                    self.writer.add_scalar(
                        "unsupervised_weight", self.unsup_weight(self.curr_epoch), itr
                    )

            # if batch_idx % 20 == 0:
            #     self.writer.add_images("train_images", image_l.cpu(), itr)
            #     self.writer.add_images("train_masks", mask.cpu(), itr)

            #     pred_arr = F.softmax(sup_preds, dim=1).cpu().detach().numpy()
            #     length = pred_arr[0].shape[1]
            #     width = pred_arr[0].shape[2]
            #     pred = np.zeros(shape=(len(pred_arr), 3, length, width))
            #     for i in range(len(pred)):
            #         pred[i, 0] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
            #         pred[i, 1] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
            #         pred[i, 2] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
            #     self.writer.add_images("predicted_masks", pred, itr)

            if mode == "semi":
                tbar.set_description(
                    "train epoch {} | s: {:.2f} us: {:.2f} |".format(
                        self.curr_epoch, s_loss, us_loss
                    )
                )
            else:
                tbar.set_description(
                    "train epoch {} | s: {:.2f} us: NA |".format(
                        self.curr_epoch, s_loss
                    )
                )

    def _valid_epoch(self):
        self.model.eval()
        dataloader = iter(self.valid_dataloader)

        term_size = shutil.get_terminal_size()
        tbar = tqdm(range(len(self.valid_dataloader)), ncols=term_size.columns)

        ave_TPR, ave_FPR, ave_IoU = 0, 0, 0
        with torch.no_grad():
            for _ in tbar:
                image, mask = next(dataloader)
                image = image.cuda(device=self.device, non_blocking=True)
                mask = mask.cuda(device=self.device, non_blocking=True)

                preds = self.model(image)
                # loss = self.sup_loss_fn(preds, mask.squeeze(1).long())
                TPR, FPR, IoU = 0, 0, 0
                for i in range(preds.shape[0]):
                    predi = preds.cpu().numpy()[i][1] > preds.cpu().numpy()[i][0]
                    TPRi, FPRi, IoUi, _, _ = utils.qScore(
                        predi, mask.squeeze(1).cpu().numpy()[i], cat="pixel"
                    )
                    TPR += TPRi
                    FPR += FPRi
                    IoU += IoUi

                ave_TPR += TPR / preds.shape[0]
                ave_FPR += FPR / preds.shape[0]
                ave_IoU += IoU / preds.shape[0]

                tbar.set_description(
                    "valid epoch {} |  iou: {:.2f} |".format(
                        self.curr_epoch, IoU / preds.shape[0]
                    )
                )
        ave_TPR /= len(self.valid_dataloader)
        ave_FPR /= len(self.valid_dataloader)
        ave_IoU /= len(self.valid_dataloader)

        print("validation iou: {:.2f}".format(ave_IoU))
        # self.writer.add_scalar("valid_tpr", ave_TPR, self.curr_epoch)
        # self.writer.add_scalar("valid_fpr", ave_FPR, self.curr_epoch)
        self.writer.add_scalar("valid_iou", ave_IoU, self.curr_epoch)

    def _example_image(self, images, masks, prefix="img"):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(images.cuda(device=self.device, non_blocking=True))

            for b in range(len(preds)):
                mask = masks[b].cpu().numpy().squeeze()
                image = images[b].cpu().numpy().transpose(1, 2, 0)
                pred_arr = F.softmax(preds[b], dim=0).cpu().numpy()
                pred = utils.PostProcessing(pred_arr[1] > pred_arr[0])

                fig = plt.figure(figsize=(9, 4))
                ax1 = plt.subplot(1, 3, 1)
                ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
                ax3 = plt.subplot(1, 3, 3)

                ax1.imshow(image)
                ax1.set_axis_off()
                ax1.set_title("Input Image")

                ax2.imshow(mask)
                ax2.set_axis_off()
                ax2.set_title("Ground Truth Mask")

                ax3.imshow(pred)
                ax3.set_axis_off()
                ax3.set_title("Predicted Mask")

                plt.savefig(f"./out/{self.name}/{prefix}_{self.curr_epoch}-{b}.png")
                plt.close()

    def _get_available_devices(self, n_gpu):
        pynvml.nvmlInit()
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print("No GPUs detected, using the CPU")
            return "cpu", 0

        elif n_gpu > sys_gpu:
            print(
                f"Number of GPU requested is {n_gpu} but only {sys_gpu} are available"
            )
            n_gpu = sys_gpu

        free_gpus = []
        for id in range(sys_gpu):
            h = pynvml.nvmlDeviceGetHandleByIndex(id)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            # print(id, info)
            # if the memory is more than 97.5% unused, we'll assume the GPU is free.
            if (info.free / info.total) > 0.975:
                free_gpus.append(id)

        device = torch.device(f"cuda:{free_gpus[0]}")
        print(
            f"Unccoupied GPUs: {len(free_gpus)} Requested: {n_gpu} Running on (ids): {free_gpus[:n_gpu]}"
        )
        if len(free_gpus) == 0:
            raise SystemError("No Available GPUs")
        gpu_ids = free_gpus[:n_gpu]
        return device, gpu_ids

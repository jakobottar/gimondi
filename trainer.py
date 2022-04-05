import torch
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
import pynvml
from torchvision.transforms import ColorJitter, RandomPerspective
from torchvision.transforms.functional import rotate, perspective
import utils
import namegenerator


def plot(imgs, row_title=None, **imshow_kwargs):
    # if not isinstance(imgs[0], list):
    #     # Make a 2d grid even if there's just 1 row
    #     imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


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
        self.unsup_loss_fn = torch.nn.CrossEntropyLoss()

        if name:
            self.name = name
        else:
            self.name = namegenerator.gen()

        self.writer = SummaryWriter("runs/" + self.name)
        print(f"run name: {self.name}")
        os.makedirs(f"./out/{self.name}/", exist_ok=True)

        self.curr_epoch = 0
        self.unsup_weight = lambda e: min(e * 1e-4, 0.03)

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
            return "super"

        print(f"training for {epochs} epochs: ")
        for e in range(epochs):
            self.curr_epoch = e
            self._train_epoch(mode=schedule(e))
            # self._train_epoch(mode="super")
            (images, masks), (_, _) = next(iter(self.train_dataloader))
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
            (image_l, mask), (image_ul, mask_ul) = next(dataloader)
            image_l = image_l.cuda(device=self.device, non_blocking=True)
            mask = mask.cuda(device=self.device, non_blocking=True)

            # Compute prediction error
            sup_preds = self.model(image_l)
            loss = self.sup_loss_fn(sup_preds, mask.squeeze(1).long())
            s_loss = loss.item()

            # unsupervised section
            if mode == "semi":

                if batch_idx % 100 == 0:
                    images = [None] * 4

                def greaterthan(x):
                    out = torch.ones((3,1,512,512),dtype = int, device = x.device)
                    for img in range(len(x)):
                        out[img] = x[img][1] > x[img][0]
                    return out

                def makebinary(mask):
                    return (
                        mask.detach().cpu().numpy()[1] > mask.detach().cpu().numpy()[0]
                    )

                def unsup_iter(image, func, target, out_arr, idx):
                    image = func(image)
                    if out_arr:
                        out_arr[idx] = [
                            image[0].detach().cpu().numpy().transpose(1, 2, 0),
                            None,
                        ]
                    image = self.model(image)
                    if out_arr:
                        out_arr[idx][1] = makebinary(F.softmax(image[0], dim=0))

                    return self.unsup_loss_fn(image, func(target.squeeze(1))), out_arr

                image_ul = image_ul.cuda(device=self.device, non_blocking=True)

                # make target mask
                target = image_ul.clone().detach()
                target = self.model(target)
                target = greaterthan(F.softmax(target, dim=1))
                # target = mask_ul.to(image_ul.device)

                if batch_idx % 100 == 0:
                    images[0] = [
                        image_ul[0].detach().cpu().numpy().transpose(1, 2, 0),
                        mask_ul[0].detach().cpu().numpy().squeeze(),
                    ]
                    images[1] = [
                        image_ul[0].detach().cpu().numpy().transpose(1, 2, 0),
                        # makebinary(target[0]),
                        target[0].detach().cpu().numpy().squeeze(),
                    ]

                # rotation
                rot = np.random.randint(0, 4) * 90
                us_loss, images = unsup_iter(
                    image_ul.clone().detach(),
                    lambda x: rotate(x, rot),
                    target,
                    images,
                    2,
                )

                # perspective warp
                pwarp = RandomPerspective().get_params(
                    target.shape[2], target.shape[3], 0.25
                )
                pw_loss, images = unsup_iter(
                    image_ul.clone().detach(),
                    lambda x: perspective(x, pwarp[0], pwarp[1]),
                    target,
                    images,
                    3,
                )
                us_loss += pw_loss

                if batch_idx % 100 == 0:
                    plot(images, ["Original", "Target", "Rotation", "Persp. Warp"])
                    plt.savefig(
                        f"./out/{self.name}/unsup_masks_{self.curr_epoch}_{batch_idx}.png"
                    )
                    plt.close()

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
                        predi, mask.squeeze(1).cpu().numpy()[i], cat="particle"
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

        print(
            "validation tpr: {:.2f}, fpr: {:.2f}, iou: {:.2f}".format(
                ave_TPR, ave_FPR, ave_IoU
            )
        )
        self.writer.add_scalar("valid_tpr", ave_TPR, self.curr_epoch)
        self.writer.add_scalar("valid_fpr", ave_FPR, self.curr_epoch)
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

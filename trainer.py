from sys import prefix
from torch import tensor
from matplotlib import image
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

from torch.distributions.uniform import Uniform

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
        self.sup_loss = loss
        self.unsup_loss = torch.nn.MSELoss()
        if name:
            self.name = name
        else:
            self.name = namegenerator.gen()
        self.writer = SummaryWriter("runs/" + self.name)
        print(f"run name: {self.name}")
        os.makedirs(f"./out/{self.name}/", exist_ok=True)

        self.mode = mode
        print(f"training mode: {self.mode}")

        # set device
        self.device, gpu_ids = self._get_available_devices(n_gpu)
        # TODO: migrate to DistributedDataParallel
        # TODO: this breaks if I pass in a pretrained dataparallel model.
        # Change to saving model.module.state_dict() then create model and load weights.
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)
        print(self.device)
        self.model.to(self.device)

        self.curr_epoch = 0

        self.uniform = Uniform(-0.25, 0.25)
        self.weight_schedule = lambda e: min(1e-3 * e, 0.05)

    def train(self, epochs: int):
        print(f"training for {epochs} epochs: ")
        for e in range(epochs):
            self.curr_epoch = e
            self._train_epoch()
            (images, masks), _ = next(iter(self.train_dataloader))
            self._example_image(images, masks, prefix="train")

            self._valid_epoch()
            images, masks = next(iter(self.valid_dataloader))
            self._example_image(images, masks, prefix="valid")

            torch.save(
                self.model.module.state_dict(), f"./out/{self.name}/checkpoint_{e}.pth"
            )
        torch.save(self.model.module.state_dict(), f"./out/{self.name}/model.pth")
        self.writer.close()

    def _train_epoch(self):
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
            loss = self.sup_loss(sup_preds, mask.squeeze(1).long())
            sup_loss = loss.item()

            if self.mode == "semi":
                image_ul = image_ul.cuda(device=self.device, non_blocking=True)
                target_ul = self.model(image_ul)
                target_ul = F.softmax(target_ul.detach(), dim=1)

                ### Additive Noise
                noisy = image_ul.clone().detach()
                for i in range(len(noisy)):
                    noise = (
                        self.uniform.sample(noisy[i].shape[1:])
                        .to(noisy.device)
                        .unsqueeze(0)
                    ) * torch.max(noisy[i].detach())
                    noisy[i] += noise

                unsup_preds = self.model(noisy)
                addnoise_loss = self.unsup_loss(unsup_preds, target_ul)

                # ### Brightness Jitter
                jitter = ColorJitter(saturation=0, hue=0)
                color = jitter(image_ul.clone().detach())

                unsup_preds = self.model(color)
                coljitter_loss = self.unsup_loss(unsup_preds, target_ul)

                unsup_loss = coljitter_loss + addnoise_loss
                loss += self.weight_schedule(self.curr_epoch) * unsup_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                itr = (self.curr_epoch * len(dataloader)) + batch_idx
                if self.mode == "semi":
                    self.writer.add_scalar(
                        "additive_noise_loss", addnoise_loss.item(), itr
                    )
                    self.writer.add_scalar(
                        "color_jitter_loss", coljitter_loss.item(), itr
                    )
                    self.writer.add_scalar(
                        "unsup_weight", self.weight_schedule(self.curr_epoch), itr
                    )

                self.writer.add_scalar("supervised_loss", sup_loss, itr)

            # TODO: fix!
            if batch_idx % 20 == 0:
                self.writer.add_images("train_images", image_l.cpu(), itr)
                self.writer.add_images("train_masks", mask.cpu(), itr)

                pred_arr = F.softmax(sup_preds, dim=1).cpu().detach().numpy()
                length = pred_arr[0].shape[1]
                width = pred_arr[0].shape[2]
                pred = np.zeros(shape=(len(pred_arr), 3, length, width))
                for i in range(len(pred)):
                    pred[i, 0] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                    pred[i, 1] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                    pred[i, 2] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                self.writer.add_images("predicted_masks", pred, itr)

                # if self.mode == "semi":
                # self.writer.add_images("unsupervised_images", image_ul.cpu(), itr)

                # pred_arr = target_ul.cpu().numpy()
                # length = pred_arr[0].shape[1]
                # width = pred_arr[0].shape[2]
                # pred = np.zeros(shape=(len(pred_arr), 3, length, width))
                # for i in range(len(pred)):
                #     pred[i, 0] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                #     pred[i, 1] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                #     pred[i, 2] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                # self.writer.add_images("unsupervised_target_masks", pred, itr)

                # pred_arr = F.softmax(unsup_preds, dim=1).cpu().detach().numpy()
                # length = pred_arr[0].shape[1]
                # width = pred_arr[0].shape[2]
                # pred = np.zeros(shape=(len(pred_arr), 3, length, width))
                # for i in range(len(pred)):
                #     pred[i, 0] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                #     pred[i, 1] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                #     pred[i, 2] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                # self.writer.add_images("unsupervised_predicted_masks", pred, itr)

            if self.mode == "semi":
                tbar.set_description(
                    "train epoch {} | s: {:.2f} us: {:.2f} |".format(
                        self.curr_epoch, sup_loss, unsup_loss.item()
                    )
                )
            else:
                tbar.set_description(
                    "train epoch {} | s: {:.2f} us: NA |".format(
                        self.curr_epoch, sup_loss
                    )
                )

    def _valid_epoch(self):
        self.model.eval()
        dataloader = iter(self.valid_dataloader)

        term_size = shutil.get_terminal_size()
        tbar = tqdm(range(len(self.valid_dataloader)), ncols=term_size.columns)

        ave_loss = 0
        with torch.no_grad():
            for batch_idx in tbar:
                image, mask = next(dataloader)
                image, mask = image.cuda(
                    device=self.device, non_blocking=True
                ), mask.cuda(device=self.device, non_blocking=True)

                preds = self.model(image)
                loss = self.sup_loss(preds, mask.squeeze(1).long())
                ave_loss += loss.mean().item()

                tbar.set_description(
                    "valid epoch {} | loss: {:.2f} |".format(
                        self.curr_epoch, loss.mean().item()
                    )
                )
        ave_loss /= len(self.valid_dataloader)

        print("validation error: {:.2f}".format(ave_loss))
        self.writer.add_scalar("valid_loss", ave_loss, self.curr_epoch)

    def _example_image(self, images, masks, prefix="img"):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(images.cuda(device=self.device, non_blocking=True))

            for b in range(len(preds)):
                mask = masks[b].cpu().numpy().squeeze()
                image = images[b].cpu().numpy().transpose(1, 2, 0)
                pred_arr = F.softmax(preds[b], dim=0).cpu().numpy()
                pred = pred_arr[1] > pred_arr[0]

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
            #! this is giving me wierd info...
            h = pynvml.nvmlDeviceGetHandleByIndex(id)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            # print(id, info)
            # if the memory is more than 97.5% unused, we'll assume the GPU is free.
            if (info.free / info.total) > 0.975:
                free_gpus.append(id)

        device = torch.device(f"cuda:{free_gpus[0]}")
        print(device)
        print(
            f"Unccoupied GPUs: {len(free_gpus)} Requested: {n_gpu} Running on (ids): {free_gpus[:n_gpu]}"
        )
        if len(free_gpus) == 0:
            raise SystemError("No Available GPUs")
        gpu_ids = free_gpus[:n_gpu]
        # return device, gpu_ids
        return torch.device("cuda:1"), [1, 2]

from sys import prefix
import torch
import os
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
import pynvml

import namegenerator

class Trainer:
    def __init__(self, model, train_dataloader, validation_dataloader, optimizer, n_gpu = 2, mode = 'semi', name = 'None') -> None:
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader
        self.optimizer = optimizer
        if name != "None":
            self.name = name
        else:
            self.name = namegenerator.gen()
        self.writer = SummaryWriter('runs/'+self.name)
        print(f"run name: {self.name}")
        os.makedirs(f"./out/{self.name}/", exist_ok=True)
        
        self.mode = mode
        print(f"training mode: {self.mode}")

        # (image_l, mask), image_ul  = next(iter(self.dataloader))
        # self.writer.add_graph(self.model, (image_l, mask, image_ul)) 
        
        # set device
        self.device, gpu_ids = self._get_available_devices(n_gpu)
        self.model = torch.nn.DataParallel(self.model, device_ids=gpu_ids)
        self.model.to(self.device)

        self.curr_epoch = 0

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

            torch.save(self.model, f'./out/{self.name}/checkpoint_{e}.pth')
        torch.save(self.model, f'./out/{self.name}/model.pth')
        self.writer.close()
        
    def _train_epoch(self):
        self.model.train()
        dataloader = iter(self.train_dataloader) # we need to do this each time to "reset" the iter obj.

        term_size = shutil.get_terminal_size()
        tbar = tqdm(range(len(self.train_dataloader)), ncols=term_size.columns)

        for batch_idx in tbar:
            (image_l, mask), image_ul = next(dataloader)
            image_l= image_l.cuda(device=self.device, non_blocking=True)
            mask = mask.cuda(device=self.device, non_blocking=True)
            if self.mode == 'semi': image_ul = image_ul.cuda(device=self.device, non_blocking=True)

            # Compute prediction error
            preds, loss = self.model(image_l, mask, image_ul) 

            # Backpropagation
            self.optimizer.zero_grad()
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                itr = (self.curr_epoch * len(dataloader)) + batch_idx
                self.writer.add_scalar("total_loss", loss.item(), itr)
                # if self.mode == 'semi': 
                #     self.writer.add_scalar("supervised_loss", loss_vals["main"], itr)
                #     self.writer.add_scalar("dropout_loss", loss_vals["dropout"], itr)
                #     self.writer.add_scalar("noisy_loss", loss_vals["noisy"], itr)

                self.writer.add_images('train_images', image_l.cpu(), itr)
                self.writer.add_images('train_masks', mask.cpu(), itr)
                
                pred_arr = F.softmax(preds[0], dim=1).cpu().detach().numpy()
                pred = np.zeros((len(pred_arr), 3, pred_arr[0].shape[1], pred_arr[0].shape[2])) 
                for i in range(len(pred)):
                    pred[i, 0] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                    pred[i, 1] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                    pred[i, 2] = (pred_arr[i, 1] > pred_arr[i, 0]).astype(int)
                self.writer.add_images('predicted_masks', pred, itr)

            tbar.set_description("train epoch {} | loss: {:.2f} |".format(self.curr_epoch, loss.item()))
    
    def _valid_epoch(self):
        self.model.eval()
        dataloader = iter(self.valid_dataloader)

        term_size = shutil.get_terminal_size()
        tbar = tqdm(range(len(self.valid_dataloader)), ncols=term_size.columns)

        ave_loss = 0
        with torch.no_grad():
            for batch_idx in tbar:
                image, mask = next(dataloader)
                image, mask = image.cuda(device=self.device, non_blocking=True), \
                    mask.cuda(device=self.device, non_blocking=True)

                _, loss = self.model(image, mask)
                ave_loss += loss.mean().item()
                
                tbar.set_description("valid epoch {} | loss: {:.2f} |".format(self.curr_epoch, loss.mean().item()))
        ave_loss /= len(self.valid_dataloader)

        print("validation error: {:.2f}".format(ave_loss))
        self.writer.add_scalar("valid_loss", ave_loss, self.curr_epoch)

    def _example_image(self, images, masks, prefix = "img"):
        self.model.eval()
        with torch.no_grad():
            preds, _ = self.model(images.cuda(device=self.device, non_blocking=True), masks.cuda(device=self.device, non_blocking=True))
            preds = preds[0]

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
                ax1.set_title('Input Image')

                ax2.imshow(mask)
                ax2.set_axis_off()
                ax2.set_title("Ground Truth Mask")

                ax3.imshow(pred)
                ax3.set_axis_off()
                ax3.set_title('Predicted Mask')

                plt.savefig(f'./out/{self.name}/{prefix}_{self.curr_epoch}-{b}.png')
                plt.close()

    def _get_available_devices(self, n_gpu): #TODO: re-check in-between epochs
        pynvml.nvmlInit()
        sys_gpu = torch.cuda.device_count()
        if sys_gpu == 0:
            print('No GPUs detected, using the CPU')
            return 'cpu', 0

        elif n_gpu > sys_gpu:
            print(f'Number of GPU requested is {n_gpu} but only {sys_gpu} are available')
            n_gpu = sys_gpu

        free_gpus = []
        for id in range(sys_gpu):
            h = pynvml.nvmlDeviceGetHandleByIndex(id)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            if (info.free / info.total) > 0.975: # if the memory is more than 97.5% unused, we'll assume the GPU is free. 
                free_gpus.append(id)

        device = torch.device(f'cuda:{free_gpus[0]}')
        print(f'Unccoupied GPUs: {len(free_gpus)} Requested: {n_gpu} Running on (ids): {free_gpus[:n_gpu]}')
        if len(free_gpus) == 0:
            raise SystemError("No Available GPUs")
        gpu_ids = free_gpus[:n_gpu]
        return device, gpu_ids

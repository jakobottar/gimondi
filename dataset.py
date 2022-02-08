import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
from skimage import io


class SegmentationImageDataset(Dataset):
    def __init__(self, dataset_file, transform=None, target_transform=None):
        self.dataset = pd.read_csv(dataset_file)
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512))])
        self.target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512))])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset.iloc[idx]['image_file']
        image = io.imread(image_path)
        mask_path = self.dataset.iloc[idx]['mask_file']
        mask = io.imread(mask_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


class UnsupervisedSegmentationDataset(SegmentationImageDataset):
    def __getitem__(self, idx):
        image_path = self.dataset.iloc[idx, 0]
        image = io.imread(image_path)
        if self.transform:
            image = self.transform(image)
        return image
        
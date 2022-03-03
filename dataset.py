from logging.handlers import RotatingFileHandler
import random
from utils import Cutout
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
from skimage import io


class SegmentationImageDataset(Dataset):
    def __init__(self, dataset_file, cutout=False, rotate=False, flip=False):
        self.dataset = pd.read_csv(dataset_file)
        # self.cutout = cutout
        self.rotate = rotate
        self.flip = flip

    def transform(self, image, mask):
        # n_holes = 2
        # length = 128

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask)

        # if self.cutout:
        #     # get random cutout mask (to maintain consistency between image and mask)
        #     cutout_mask = Cutout.get_params(image, n_holes, length)
        #     image = image * cutout_mask
        #     mask = mask * cutout_mask

        if self.rotate:
            ang = random.randint(0, 4) * 90
            image = transforms.functional.rotate(image, ang)
            mask = transforms.functional.rotate(mask, ang)

        if self.flip:
            if random.random() > 0.5:  # horizontal flip
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

            if random.random() > 0.5:  # vertical flip
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)

        resize = transforms.Resize((512, 512))
        image = resize(image)
        mask = resize(mask)

        return image, mask

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset.iloc[idx]["image_file"]
        image = io.imread(image_path)
        mask_path = self.dataset.iloc[idx]["mask_file"]
        mask = io.imread(mask_path)
        return self.transform(image, mask)


class UnsupervisedSegmentationDataset(SegmentationImageDataset):
    def transform(self, image):
        # n_holes = 2
        # length = 128

        to_tensor = transforms.ToTensor()
        image = to_tensor(image)

        # if self.cutout:
        #     # get random cutout mask (to maintain consistency between image and mask)
        #     cutout = Cutout(n_holes, length)
        #     image = cutout(image)

        if self.rotate:
            ang = random.randint(0, 4) * 90
            image = transforms.functional.rotate(image, ang)

        if self.flip:
            if random.random() > 0.5:  # horizontal flip
                image = transforms.functional.hflip(image)

            if random.random() > 0.5:  # vertical flip
                image = transforms.functional.vflip(image)

        resize = transforms.Resize((512, 512))
        image = resize(image)

        return image

    def __getitem__(self, idx):
        image_path = self.dataset.iloc[idx, 0]
        image = io.imread(image_path)
        return self.transform(image)

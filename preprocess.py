
import os
import csv
import random
import numpy as np
import pandas as pd
from PIL import Image
import skimage.measure as skiM

import shutil
from tqdm import tqdm

def randomPatch(image_width, image_height, patch_width, patch_height):
    x1, y1 = 0, 0
    if image_width-patch_width != 0 or image_height-patch_height != 0:
        x1 = random.randrange(0, image_width-patch_width)
        y1 = random.randrange(0, image_height-patch_height)

    return (x1, y1, x1+patch_width, y1+patch_height)

# Extract color pixels as segmented pixels
# TODO: Is there a faster way to do this? this takes a long time
def makeMask(img):
    img_arr = np.array(img)

    r = img_arr[:, :, 0].reshape(img_arr.shape[0], img_arr.shape[1])
    g = img_arr[:, :, 1].reshape(img_arr.shape[0], img_arr.shape[1])
    b = img_arr[:, :, 2].reshape(img_arr.shape[0], img_arr.shape[1])		

    temp1 = np.zeros((r.shape)); temp1[np.where(r != g)] = 1
    temp2 = np.zeros((r.shape)); temp2[np.where(b != g)] = 1
    temp3 = np.zeros((r.shape)); temp3[np.where(r != b)] = 1

    t = temp1 + temp2 + temp3
    labelImg = np.zeros((r.shape))
    labelImg[np.where(t > 1)] = 255

    if True:
        temp = np.zeros((r.shape)); temp[np.where(g == 255)] = 1			
        labelImg[np.where(temp > 0)] = 0					
        
    return Image.fromarray(labelImg).convert(mode="RGB")

# Remove regions near boundaries
def boundaryFilt(image):

	label = (np.array(image) / 255).astype('uint8')

	topCut = 2; endCut = -1
	pLabel = np.copy(label)
	
	temp = np.copy(label)
	temp[:,:topCut] = np.max(label)
	temp[:,endCut:] = np.max(label)
	temp[:topCut,:] = np.max(label)
	temp[endCut:,:] = np.max(label)	

	pCount = skiM.label(temp, background = 0) + 1	
	if np.min(pCount) != 0:
		pCount -= 1

	pCount[np.where(pCount == 1)] = 0

	pLabel[np.where(pCount == 0)] = 0
	
	return Image.fromarray(pLabel * 255)

def Preprocess(dataset, csvwriter, num_patches, height, width, num_ang, train_file_location = "./data/train/", prefix = "uo3") -> None:
    r"""This function preprocesses a set of images and corresponding masks for training.
    It reads in the images and masks and performs random crops and rotations on the images
    args:
        image_source: the source filepath for the image files
        mask_source: the source filepath for the corresponding masks
        num_patches: how many random patches of the image we should crop
        height: height of cropped patches
        width: width of cropped patches
        num_ang: number of 90-degree rotations to generate
            rotations are in order, 0, 90, 180, 270
    """
    ext_length = 4 # length of the file extension

    # get list if images and masks from chosen folders

    term_size = shutil.get_terminal_size()
    tbar = tqdm(range(len(dataset)), ncols=term_size.columns)

    for i in tbar:
        image_path = dataset.iloc[i]['image_file']
        mask_path = dataset.iloc[i]['mask_file']
        if mask_path == "None": mask_path = False

        filename = image_path.split('/')[-1][:-ext_length] # get filename
        # ext = img_file.split('/')[-1][-ext_length:] # get file extension
        ext = ".png"
        
        # load individual image files
        img_raw = Image.open(image_path)
        if mask_path: msk_raw = Image.open(mask_path)

        # our raw images contain a legend at the bottom which needs to be cropped off
        img = img_raw.crop((0,0,1024,880))
        if mask_path: msk = msk_raw.crop((0,0,1024,880))

        # convert the mask color image to a binary mask
        if mask_path: msk = makeMask(msk)

        # create and save paths for saving mask and image files
        new_image_path = os.path.join(train_file_location, "images")
        if mask_path: new_mask_path = os.path.join(train_file_location, "masks")

        try: os.makedirs(new_image_path) # create image folder if it does not exist
        except FileExistsError: True
        try: os.makedirs(new_mask_path) # create mask folder if it does not exist
        except FileExistsError: True

        # get random patches and rotate them
        for p in range(num_patches):
            # generate a random patch of the original image 
            # and crop image and mask with same patch
            img_w, img_h = img.size
            patch = randomPatch(img_w, img_h, width, height)
            img_crop = img.crop(patch)
            if mask_path: msk_crop = msk.crop(patch)

            for a in range(num_ang):
                # rotate images and masks
                ang = (a % 4) * 90
                img_rot = img_crop.rotate(ang)			
                if mask_path: msk_rot = msk_crop.rotate(ang)

                if mask_path: msk_filt = boundaryFilt(msk_rot)

                # generate filenames for image and mask
                img_name = os.path.join(new_image_path, f"{prefix}_{i}_{p}_{a}.png")
                if mask_path: mask_name = os.path.join(new_mask_path, f"{prefix}_{i}_{p}_{a}_mask.gif") 

                if mask_path: csvwriter.writerow([img_name, mask_name])
                else: csvwriter.writerow([img_name, "None"])

                # save files
                img_rot.save(img_name) 
                if mask_path: msk_filt.save(mask_name)

        tbar.set_description("image {} |".format(i))

if __name__=="__main__":
    dataset_file = "/home/sci/jakobj/datasets/alpha-uo3/alpha-uo3.csv"
    dataset = pd.read_csv(dataset_file)

    sup_ix = 10
    unsup_ix = 50

    dataset.sample(frac=1).reset_index(drop=True)

    supervised = dataset.iloc[0:sup_ix]
    unsupervised = dataset.iloc[sup_ix:unsup_ix]
    validation = dataset.iloc[50:]

    # unsegmented_file = "/home/sci/jakobj/datasets/uo3-unsegmented/files.csv"
    # unseg_dataset = pd.read_csv(unsegmented_file)

    # unsupervised = pd.concat([unsupervised, unseg_dataset], ignore_index=True)

    try: os.makedirs("./temp/")
    except FileExistsError: None

    # feed into preprocessor
    with open('./temp/test.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['image_file', 'mask_file'])
        Preprocess(validation, csvwriter, 30, 512, 512, 4, "/scratch/jakobj/nfs/", "valid_uo3")

    with open('./temp/train_supervised.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['image_file', 'mask_file'])
        Preprocess(supervised, csvwriter, 30, 512, 512, 4, "/scratch/jakobj/nfs/", "sup_uo3")

    with open('./temp/train_unsupervised.csv', 'w') as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(['image_file', 'mask_file'])
        Preprocess(unsupervised, csvwriter, 30, 512, 512, 4, "/scratch/jakobj/nfs/", "unsup_uo3")

    print("finished preprocessing raw images.")

## script to generate testing/training text files from segmentation datasets
import argparse
import os
import numpy as np
import csv
import math

#TODO: merge into preprocess.py

parser = argparse.ArgumentParser()
# /usr/sci/projs/DeepLearning/Jakob_Dataset/plant-segmentation/plants
parser.add_argument('-d', '--directory',type=str, default='./images/', help="dataset location")
parser.add_argument('--images',type=str, default='images', help="image location")
parser.add_argument('--masks',type=str, default='masks', help="image location")
parser.add_argument('--test-split',type=float, default=0.25, help="percentage of dataset set aside for testing (0..1)")
parser.add_argument('--supervised-split',type=float, default=0.5, help="percentage of the dataset used for supervised learning")
FLAGS = parser.parse_args()

np.random.seed(33)

images_dir = os.path.join(FLAGS.directory, FLAGS.images)
masks_dir = os.path.join(FLAGS.directory, FLAGS.masks)

images_files = list(map(lambda f : os.path.join(images_dir, f), os.listdir(images_dir)))
images_files.sort()
masks_files = list(map(lambda f : os.path.join(masks_dir, f), os.listdir(masks_dir)))
masks_files.sort()

files = np.column_stack([images_files, masks_files])
np.random.shuffle(files)
test_split = int(FLAGS.test_split * len(files))

try: os.makedirs("./temp/")
except FileExistsError: None

with open('./temp/test.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['image_file', 'mask_file'])
    csvwriter.writerows(files[:test_split])

train_set = files[test_split:]
num_sup = math.floor(len(train_set) * FLAGS.supervised_split)

with open('./temp/train_supervised.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['image_file', 'mask_file'])
    csvwriter.writerows(train_set[:num_sup])

with open('./temp/train_unsupervised.csv', 'w') as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(['image_file', 'mask_file'])
    csvwriter.writerows(train_set[num_sup:])

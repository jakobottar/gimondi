import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
from models import UNet
import trainer
import utils

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of training epochs (default: 5)" )
    parser.add_argument("-g", "--n-gpus", type=int, default=2, help="max number of GPUs to use (default: 2)" )
    parser.add_argument("--mode", type=str, default='semi', help="model mode (default: semi)" )
    parser.add_argument("--name", type=str, default='None', help="model name (default: random name)")
    FLAGS = parser.parse_args()

    test_dataset = dataset.SegmentationImageDataset("./temp/test.csv")
    sup_dataset = dataset.SegmentationImageDataset("./temp/train_supervised.csv")
    unsup_dataset = dataset.UnsupervisedSegmentationDataset("./temp/train_unsupervised.csv")

    print(f"datasets loaded, {len(sup_dataset)} labeled examples, {len(unsup_dataset)} unlabeled examples")

    batch_size = 10
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    sup_dataloader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_dataloader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    dataloader = utils.SemiSupervisedDataLoader(sup_dataloader, unsup_dataloader, FLAGS.mode)

    net = UNet(sup_loss="crossentropy", unsup_loss="mse", mode=FLAGS.mode)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0.1)

    net_trainer = trainer.Trainer(net, dataloader, test_dataloader, optimizer, FLAGS.n_gpus, FLAGS.mode, FLAGS.name)
    net_trainer.train(epochs=FLAGS.epochs)

    print("done!")
    
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
from models import UNet
import trainer
import utils

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5,
        help="number of training epochs (default: 5)",
    )
    parser.add_argument(
        "-g",
        "--n-gpus",
        type=int,
        default=2,
        help="max number of GPUs to use (default: 2)",
    )
    parser.add_argument(
        "--mode", type=str, default="semi", help="model mode (default: semi)"
    )
    parser.add_argument(
        "--name", type=str, default=None, help="model name (default: random name)"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="add a pretrained model here"
    )
    FLAGS = parser.parse_args()

    test_dataset = dataset.SegmentationImageDataset("./data/test.csv")
    sup_dataset = dataset.SegmentationImageDataset(
        "./data/train_supervised.csv", rotate=True, flip=True
    )
    unsup_dataset = dataset.UnsupervisedSegmentationDataset(
        "./data/train_unsupervised.csv", rotate=True, flip=True
    )

    print(
        f"datasets loaded:\n    {len(sup_dataset)} labeled examples\n    {len(unsup_dataset)} unlabeled examples\n    {len(test_dataset)} testing examples"
    )

    batch_size = 16
    test_dataloader = DataLoader(test_dataset, batch_size=4)
    sup_dataloader = DataLoader(sup_dataset, batch_size=batch_size, shuffle=True)
    unsup_dataloader = DataLoader(unsup_dataset, batch_size=batch_size, shuffle=True)
    dataloader = utils.SemiSupervisedDataLoader(
        sup_dataloader, unsup_dataloader, FLAGS.mode
    )

    if FLAGS.pretrained:
        net = torch.load(FLAGS.pretrained)
    else:
        net = UNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01, weight_decay=0.1)
    loss = nn.CrossEntropyLoss()

    net_trainer = trainer.Trainer(
        net,
        dataloader,
        test_dataloader,
        loss,
        optimizer,
        FLAGS.n_gpus,
        FLAGS.mode,
        FLAGS.name,
    )
    net_trainer.train(epochs=FLAGS.epochs)

    print("done!")

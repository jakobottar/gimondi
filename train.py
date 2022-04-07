import argparse
import json
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
        "-c",
        "--config",
        type=str,
        default="./config.json",
        help="config file location (default: config.json)",
    )
    parser.add_argument(
        "-n", "--name", type=str, default=None, help="model name (default: random name)"
    )
    FLAGS = parser.parse_args()

    with open(FLAGS.config) as file:
        config = json.load(file)

    test_dataset = dataset.SegmentationImageDataset(config["data_files"]["valid"])
    sup_dataset = dataset.SegmentationImageDataset(
        config["data_files"]["labeled"], rotate=True, flip=True
    )
    if config["mode"] == "semisupervised":
        unsup_dataset = dataset.SegmentationImageDataset(
            config["data_files"]["unlabeled"], rotate=True, flip=True
        )
        len_unsup_dataset = len(unsup_dataset)
    else:
        unsup_dataset = None
        len_unsup_dataset = 0

    print(
        f"datasets loaded:\n\
    {len(sup_dataset)} labeled examples\n\
    {len_unsup_dataset} unlabeled examples\n\
    {len(test_dataset)} testing examples"
    )

    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    sup_dataloader = DataLoader(
        sup_dataset, batch_size=config["batch_size"], shuffle=True
    )
    if config["mode"] == "semisupervised":
        unsup_dataloader = DataLoader(
            unsup_dataset, batch_size=config["batch_size"], shuffle=True
        )
    else:
        unsup_dataloader = None
    dataloader = utils.SemiSupervisedDataLoader(
        sup_dataloader, unsup_dataloader, config["mode"]
    )

    net = UNet()

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    loss = nn.CrossEntropyLoss()

    net_trainer = trainer.Trainer(
        net,
        dataloader,
        test_dataloader,
        loss,
        optimizer,
        config["num_gpus"],
        config["mode"],
        FLAGS.name,
    )
    net_trainer.train(epochs=config["num_epochs"])

    print("done!")

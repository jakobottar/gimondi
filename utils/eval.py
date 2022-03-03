from ast import Return
import torch

r"""Takes in the raw output of the neural net"""


def dice_loss(pred, true):
    pred = torch.nn.functional.softmax(pred, dim=1)

    dice = []
    for i, p in enumerate(pred):
        p = p[1] > p[0]
        dice.append(dice_score(p, true[i].squeeze()))

    return torch.tensor(dice)


def dice_score(pred, true) -> torch.Tensor:
    intersection = torch.sum(pred * true)
    denominator = torch.sum(pred + true)

    dice = (2 * intersection) / (denominator)

    return 1 - dice

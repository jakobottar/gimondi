import torch
import numpy as np
import skimage.measure as skiM
import shutil


def dice_loss(pred, true):
    r"""Takes in the raw output of the neural net"""
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


## Compute quantitative scores
def qScore(prediction, label, cat="pixel"):

    if cat == "pixel":
        imgTP = []
        imgTP_label = []

        label_flat = np.copy(label).reshape(label.shape[0] * label.shape[1], 1)
        predict_flat = np.copy(prediction).reshape(
            prediction.shape[0] * prediction.shape[1], 1
        )

        # True positive labeled pixels
        indPosLabel = np.where(label_flat > 0)[0]
        # True negative labeled pixels
        indNegLabel = np.where(label_flat == 0)[0]

        # predict label pixels
        indPosPredict = np.where(predict_flat > 0)[0]

        TP = len(np.intersect1d(indPosLabel, indPosPredict))
        FP = len(np.intersect1d(indPosPredict, indNegLabel))

        TPR = 0.0
        if len(indPosLabel) > 0.0:
            TPR = TP / len(indPosLabel)

        FPR = 0.0
        if len(indNegLabel) > 0.0:
            FPR = FP / len(indNegLabel)

        IoU = 0.0
        if len(np.union1d(indPosLabel, indPosPredict)) > 0.0:
            IoU = len(np.intersect1d(indPosLabel, indPosPredict)) / len(
                np.union1d(indPosLabel, indPosPredict)
            )

    else:

        pCount = countNumRegs(prediction)
        lCount = countNumRegs(label)

        # f, ax = plt.subplots(1,2)
        # ax[0].imshow(pCount); ax[1].imshow(lCount)
        # print(pCount.shape, lCount.shape)

        lCount_flat = np.copy(lCount).reshape(lCount.shape[0] * lCount.shape[1], 1)
        pCount_flat = np.copy(pCount).reshape(pCount.shape[0] * pCount.shape[1], 1)

        # Unique regions
        valUniqueP = np.unique(pCount)
        valUniqueP = valUniqueP[np.where(valUniqueP > 0)]
        valUniqueL = np.unique(lCount)
        valUniqueL = valUniqueL[np.where(valUniqueL > 0)]

        overlapT = 0.6
        countP = 0
        countN = 0

        trueposLabel = np.zeros((label.shape))
        trueposPredict = np.copy(trueposLabel)

        for i in range(len(valUniqueP)):
            indPredict = np.where(pCount_flat == valUniqueP[i])[0]

            ## Find the region values of true label corresponding with current predict indices
            temp = np.unique(lCount_flat[indPredict])
            temp = temp[np.where(temp > 0)]

            for j in range(len(temp)):

                indTrue = np.where(lCount_flat == temp[j])[0]

                inter = np.intersect1d(indPredict, indTrue)
                un = np.union1d(indPredict, indTrue)

                if len(inter) >= overlapT * len(un):
                    countP += 1

                    ## Obtain true positive regions for both ground truth and predict
                    trueposLabel[np.where(lCount == temp[j])] = 1
                    trueposPredict[np.where(pCount == valUniqueP[i])] = 1
                else:
                    countN += 1

        imgTP = np.copy(trueposPredict)
        imgTP_label = np.copy(trueposLabel)

        trueposPredict = trueposPredict.reshape(
            trueposPredict.shape[0] * trueposPredict.shape[1], 1
        )
        indPosPredict = np.where(trueposPredict > 0)[0]
        trueposLabel = trueposLabel.reshape(
            trueposLabel.shape[0] * trueposLabel.shape[1], 1
        )
        indPosLabel = np.where(trueposLabel > 0)[0]

        # f, ax = plt.subplots(1,3)
        # ax[0].imshow(label); ax[1].imshow(prediction); ax[2].imshow(posLabel)
        numPosLabel = len(valUniqueL)
        numPosPredict = len(valUniqueP)

        TPR = 0.0
        if numPosLabel > 0.0:
            TPR = countP / numPosLabel

        FPR = 0.0
        if numPosPredict != 0.0:
            FPR = (numPosPredict - countP) / numPosPredict

        IoU = 0.0
        if len(np.union1d(indPosLabel, indPosPredict)) > 0.0:
            IoU = len(np.intersect1d(indPosPredict, indPosLabel)) / len(
                np.union1d(indPosLabel, indPosPredict)
            )

    return TPR, FPR, IoU, imgTP, imgTP_label


# Extract color pixels as segmented pixels
def labelRegions(img):
    r = img[:, :, 0].reshape(img.shape[0], img.shape[1])
    g = img[:, :, 1].reshape(img.shape[0], img.shape[1])
    b = img[:, :, 2].reshape(img.shape[0], img.shape[1])

    temp1 = np.zeros((r.shape))
    temp1[np.where(r != g)] = 1
    temp2 = np.zeros((r.shape))
    temp2[np.where(b != g)] = 1
    temp3 = np.zeros((r.shape))
    temp3[np.where(r != b)] = 1

    t = temp1 + temp2 + temp3
    labelImg = np.zeros((r.shape))
    labelImg[np.where(t > 1)] = 255

    if True:
        temp = np.zeros((r.shape))
        temp[np.where(g == 255)] = 1
        labelImg[np.where(temp > 0)] = 0

    return labelImg


# Random extracting region of image
def patchData(img, label, w=450, h=450):

    w = int(w)
    h = int(h)
    pImg = np.zeros((w, h))

    x, y = getRandCoord(w, h, img)
    pLabel = np.copy(pImg)
    if len(label) > 0:
        pLabel = np.copy(label[x : x + w, y : y + h])
        pLabel = boundaryFilt(pLabel)

        while np.max(pLabel) < 1:
            x, y = getRandCoord(w, h, img)
            pLabel = np.copy(label[x : x + w, y : y + h])
            pLabel = boundaryFilt(pLabel)

    pImg = np.copy(img[x : x + w, y : y + h])

    return pImg, pLabel


def getRandCoord(w, h, img):
    x = np.random.randint(img.shape[0])
    while x + w >= img.shape[0]:
        x = np.random.randint(img.shape[0])

    y = np.random.randint(img.shape[1])
    while y + h >= img.shape[1]:
        y = np.random.randint(img.shape[1])

    return x, y


# Remove regions near boundaries
def boundaryFilt(label):

    topCut = 2
    endCut = -1

    pLabel = np.copy(label)

    temp = np.copy(label)
    temp[:, :topCut] = np.max(label)
    temp[:, endCut:] = np.max(label)
    temp[:topCut, :] = np.max(label)
    temp[endCut:, :] = np.max(label)

    pCount = countNumRegs(temp)
    pCount[np.where(pCount == 1)] = 0

    pLabel[np.where(pCount == 0)] = 0

    return pLabel


def countNumRegs(img):

    pCount = skiM.label(img, connectivity=2, background=0) + 1
    if np.min(pCount) != 0:
        pCount -= 1
    return pCount


## Binary threshold props of regions
def propThreshold(prop, img, thres=0.7):

    new_img = np.zeros((img.shape))
    new_img[np.where(img > 0)] = 1
    mean_val = np.mean(prop)

    # plt.figure(); plt.plot(prop,'ro');
    # plt.plot(np.repeat(thres*mean_val,len(prop)),'k')
    # plt.savefig(os.getcwd() + '/thres.png')
    ind = np.where(prop < thres * mean_val)[0]

    for i in range(len(ind)):
        tempInd = np.where(img == ind[i] + 1)
        new_img[tempInd] = 0

    return new_img


## Post processing prediction
def PostProcessing(inp):
    temp = np.copy(inp)
    temp[np.where(temp > 0)] = 255
    temp = boundaryFilt(temp)

    labeled = countNumRegs(temp)
    props = skiM.regionprops(labeled)
    attr1 = [props[itr]["area"] for itr in range(len(props))]
    l1 = propThreshold(attr1, labeled, thres=0.3)  # 5)
    temp = np.copy(l1)

    if False:
        a = np.copy(temp)
        b = skiMore.opening(a, skiMore.disk(1))  # square(9))
        c = np.copy(b)
        temp = skiMore.closing(c, skiMore.disk(1))

    l1 = countNumRegs(temp)
    attr2 = [props[itr]["solidity"] for itr in range(len(props))]
    l2 = propThreshold(attr2, l1, thres=0.9)

    new_label = np.zeros((temp.shape))
    new_label[np.where(l2 > 0)] = 1.0

    return new_label

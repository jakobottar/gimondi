
from models import UNet
import utils
import dataset

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F

model_files = ["full-super.pth"]

for filename in model_files:

        model = UNet()
        model.load_state_dict(torch.load(filename))
        model.to("cuda")
        model.eval()


        test_dataset = dataset.SegmentationImageDataset("./data/valid_imgs.csv")
        test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        images, masks = next(iter(test_dataloader))

        with torch.no_grad():
            preds = model(images.cuda(device="cuda", non_blocking=True))

            for i in range(len(images)):
                ground = masks[i].cpu().numpy().squeeze()
                image = images[i].cpu().numpy().transpose(1, 2, 0)
                pred_arr = F.softmax(preds[i], dim=0).cpu().numpy()
                mask_post = utils.PostProcessing(pred_arr[1] > pred_arr[0])

                mask = Image.fromarray(mask_post.astype('bool'))
                ground_mask = Image.fromarray(ground.astype('bool')).convert(mode="1")

                source = Image.fromarray((image*255).astype('uint8')).convert(mode = "RGBA")

                image = source.copy()

                bg = Image.new(mode="RGBA", size=source.size, color="#00000000")
                
                blue_mask = bg.copy()
                blue = Image.new(mode="RGBA", size=source.size, color="#0000ff80")
                blue_mask.paste(blue, (0,0), mask)

                red_mask = bg.copy()
                red = Image.new(mode="RGBA", size=source.size, color="#ff000080")
                red_mask.paste(red, (0,0), ground_mask)

                image.save(f"./imgs/{filename}-image-{i}.png")

                image_pred = Image.alpha_composite(image, blue_mask)
                image_ground = Image.alpha_composite(image, red_mask)

                image_pred.save(f"./imgs/{filename}-image-pred-{i}.png")
                image_ground.save(f"./imgs/{filename}-image-ground-{i}.png")

    

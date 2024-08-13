import os
import argparse
import torch
import torch.nn.parallel

from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata
import pdb

from PIL import Image

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")
   

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--input_path", help="Input csv")
    parser.add_argument("--output_path", help="Output")

    args=parser.parse_args()

    if torch.cuda.device_count() == 8:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3, 4, 5, 6, 7]).cuda()
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
        batch_size = 32
    else:
        model = model.cuda()
        batch_size = 4

    semantic_weights = FCN_ResNet50_Weights.DEFAULT
    semantic_model = fcn_resnet50(weights=semantic_weights).cuda()
    semantic_preprocessor = semantic_weights.transforms()
    print("Classes", semantic_weights.meta["categories"])

    dataset_loader = loaddata.getTrainingData(args.input, batch_size)

    convert(dataset_loader, semantic_model, semantic_preprocessor=semantic_preprocessor, output_path=args.output_path, categories=semantic_weights.meta["categories"])


def convert(dataset_loader, semantic_model, semantic_preprocessor=None, output_path='data/demo/', categories=None):
    for i, sample_batched in enumerate(dataset_loader):
        image = sample_batched['image']
        semantic_input = image
        with torch.no_grad():
            if semantic_preprocessor != None:
                semantic_input = semantic_preprocessor(semantic_input)
            semantic_input = semantic_input.cuda()
            semantic_prediction = semantic_model(semantic_input)['out']
            semantic_out = semantic_prediction.softmax(dim=1)

        for i in range(semantic_out.size(1)):
            semantic = Image.fromarray(semantic_out[0][i].data.cpu().numpy())
            semantic = semantic.resize((image.size(3), image.size(2)))
            matplotlib.image.imsave(os.path.join(output_path, "semantic_{}.png".format(i if categories == None else categories[i])), semantic)

if __name__ == '__main__':
    main()

import os
import argparse
import torch
import torch.nn.parallel

from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

import loaddata

from PIL import Image, ImageColor

import pandas as pd

import matplotlib.image
import matplotlib.pyplot as plt

from configs import *
from util import get_filename_without_extension

plt.set_cmap("jet")

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--input", help="Input csv")
    parser.add_argument("--output_path", help="Output")

    args=parser.parse_args()

    if torch.cuda.device_count() == 8:
        batch_size = 64
    elif torch.cuda.device_count() == 4:
        batch_size = 32
    else:
        batch_size = 4

    semantic_weights = FCN_ResNet50_Weights.DEFAULT
    semantic_model = fcn_resnet50(weights=semantic_weights).cuda()
    semantic_preprocessor = semantic_weights.transforms()
    print("Classes", semantic_weights.meta["categories"])

    input_csv = args.input
    dataset_loader = loaddata.getTrainingData(args.input, batch_size)

    convert(input_csv, dataset_loader, semantic_model, semantic_preprocessor=semantic_preprocessor, output_path=args.output_path, categories=semantic_weights.meta["categories"])

def convert(input_csv, dataset_loader, semantic_model, semantic_preprocessor=None, output_path='data/demo/', categories=None):
    csv_content = pd.read_csv(input_csv)
    for i, sample_batched in enumerate(dataset_loader):
        base_name = get_filename_without_extension(csv_content.iloc[i, 0]).split('_')[0]
        image = sample_batched['image']
        semantic_input = image
        with torch.no_grad():
            if semantic_preprocessor != None:
                semantic_input = semantic_preprocessor(semantic_input)
            semantic_input = semantic_input.cuda()
            semantic_prediction = semantic_model(semantic_input)['out']
            semantic_out = semantic_prediction.softmax(dim=1)

        semantic_out = semantic_out[0, :, :, :]
        semantic_argmax = semantic_out.argmax(dim=0)
        semantic_colored = Image.new('RGB', (semantic_argmax.shape[1], semantic_argmax.shape[0]))
        classes_color_code_values = list(classes_color_code.values())
        print(semantic_argmax.shape)
        print(semantic_colored.size)

        for i in range(len(classes_color_code_values)):
            class_mask = (semantic_argmax == i)
            print(class_mask.shape)
            color = ImageColor.getrgb(classes_color_code_values[i])
            for j in range(semantic_colored.size[0]):
                for k in range(semantic_colored.size[1]):
                    if class_mask[k, j]:
                        semantic_colored.putpixel((j, k), color)
        
        matplotlib.image.imsave(os.path.join(output_path, "{}_semantic.png".format(base_name)), semantic_colored)
        return

if __name__ == '__main__':
    main()

import os
import argparse
import torch
import torch.nn.parallel

from torchvision.io.image import read_image
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image

from models import modules, net, resnet, densenet, senet
import numpy as np
import loaddata_demo as loaddata
import pdb

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")

def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model
   

def main():
    parser=argparse.ArgumentParser()

    parser.add_argument("--input", help="Input image")
    parser.add_argument("--output_path", help="Output folder")

    args=parser.parse_args()

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    model.eval()

    semantic_weights = FCN_ResNet50_Weights.DEFAULT
    semantic_model = fcn_resnet50(weights=semantic_weights).cuda()
    semantic_preprocessor = semantic_weights.transforms()
    print("Classes", semantic_weights.meta["categories"])

    nyu2_loader = loaddata.readNyu2(args.input)

    test(nyu2_loader, model, semantic_model, semantic_preprocessor=semantic_preprocessor, output_path=args.output_path, categories=semantic_weights.meta["categories"])


def test(nyu2_loader, model, semantic_model, semantic_preprocessor=None, output_path='data/demo/', categories=None):
    for i, image in enumerate(nyu2_loader):
        semantic_input = image
        with torch.no_grad():
            if semantic_preprocessor != None:
                semantic_input = semantic_preprocessor(semantic_input)
            semantic_input = semantic_input.cuda()
            out = model(image)
            semantic_prediction = semantic_model(semantic_input)['out']
            semantic_out = semantic_prediction.softmax(dim=1)
        matplotlib.image.imsave(os.path.join(output_path, "depth.png"), out.view(out.size(2),out.size(3)).data.cpu().numpy())
        for i in range(semantic_out.size(1)):
            matplotlib.image.imsave(os.path.join(output_path, "semantic_{}.png".format(i if categories == None else categories[i])), semantic_out[0][i]).data.cpu().numpy()

if __name__ == '__main__':
    main()

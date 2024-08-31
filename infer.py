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

from PIL import Image

import matplotlib.image
import matplotlib.pyplot as plt
plt.set_cmap("jet")
import tifffile

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
    
    # Create the output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output directory: {args.output_path}")

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load('./pretrained_model/model_senet'))
    model.eval()

    semantic_weights = FCN_ResNet50_Weights.DEFAULT
    semantic_model = fcn_resnet50(weights=semantic_weights).cuda()
    semantic_preprocessor = semantic_weights.transforms()
    print("Classes", semantic_weights.meta["categories"])

    nyu2_loader = loaddata.readNyu2(args.input)

    infer_and_save_data(nyu2_loader, model, semantic_model, semantic_preprocessor=semantic_preprocessor, output_path=args.output_path, categories=semantic_weights.meta["categories"])


def test(nyu2_loader, model, semantic_model, semantic_preprocessor=None, output_path='./data/demo/', categories=None):
    for i, image in enumerate(nyu2_loader):
        print(f"Original image size: {image.size()}")
        print(f"image size: {image.size(2), image.size(3)}")
        image_preprocessed = image
        if image_preprocessed.dim() == 4:
          image_preprocessed = image_preprocessed.squeeze(0)
        print(f"After squeeze: {image.size()}")
        if image_preprocessed.size(0) != 3:
          image_preprocessed = image_preprocessed.permute(2,0,1)
        print(f"After permute: {image.size()}")

        image_preprocessed = image_preprocessed.data.cpu().numpy().transpose(1, 2, 0)
        # Normalize to 0-255 range
        image_preprocessed = (image_preprocessed - image_preprocessed.min()) / (image_preprocessed.max() - image_preprocessed.min()) * 255
        image_preprocessed = image_preprocessed.astype(np.uint8)
        preprocessed_rgb = Image.fromarray(image_preprocessed, 'RGB')
        print(f"Preprocessed image size: {preprocessed_rgb.size}")
        
        matplotlib.image.imsave(os.path.join(output_path, "preprocessed_image.png"), image_preprocessed)
        # JAJ here is the input size for sure
        #squezzed =  image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        #preprocessed = Image.fromarray(squezzed)
        #matplotlib.image.imsave(os.path.join(output_path, "rbg_preprocessed.png"), preprocessed)

        semantic_input = image
        with torch.no_grad():
            if semantic_preprocessor != None:
                semantic_input = semantic_preprocessor(semantic_input)
            semantic_input = semantic_input.cuda()
            out = model(image)
            semantic_prediction = semantic_model(semantic_input)['out']
            semantic_out = semantic_prediction.softmax(dim=1)

        print(f"output size: {out.size()}")
        print(f"output size: {out.size(2), out.size(3)}")
        
        depth = Image.fromarray(out.view(out.size(2),out.size(3)).data.cpu().numpy())
        depth = depth.resize((image.size(3), image.size(2)))

        print(f"Depth image resize: {depth.size}")
        
        matplotlib.image.imsave(os.path.join(output_path, "depth.png"), depth)

        for i in range(semantic_out.size(1)):
            semantic = Image.fromarray(semantic_out[0][i].data.cpu().numpy())
            semantic = semantic.resize((image.size(3),image.size(2)))
            matplotlib.image.imsave(os.path.join(output_path, "semantic_{}.png".format(i if categories == None else categories[i])), semantic)
        
        print(f"Semantic image resize: {semantic.size}")
        
        
def infer_and_save_data(nyu2_loader, model, semantic_model, semantic_preprocessor=None, output_path='./data/demo/', categories=None):
    for i, image in enumerate(nyu2_loader):
        print(f"Original image size: {image.size()}")
        print(f"image size: {image.size(2), image.size(3)}")
        image_preprocessed = image
        if image_preprocessed.dim() == 4:
            image_preprocessed = image_preprocessed.squeeze(0)
        print(f"After squeeze: {image_preprocessed.size()}")
        if image_preprocessed.size(0) != 3:
            image_preprocessed = image_preprocessed.permute(2,0,1)
        print(f"After permute: {image_preprocessed.size()}")
        image_preprocessed = image_preprocessed.data.cpu().numpy().transpose(1, 2, 0)
        # Normalize to 0-255 range
        image_preprocessed = (image_preprocessed - image_preprocessed.min()) / (image_preprocessed.max() - image_preprocessed.min()) * 255
        image_preprocessed = image_preprocessed.astype(np.uint8)
        preprocessed_rgb = Image.fromarray(image_preprocessed, 'RGB')
        print(f"Preprocessed image size: {preprocessed_rgb.size}")

        # Save RGB image as numpy array
        np.save(os.path.join(output_path, f"rgb_{i}.npy"), image_preprocessed)
        print(f"Saved RGB image with shape: {image_preprocessed.shape}")

        # Save visualizable version
        preprocessed_rgb.save(os.path.join(output_path, f"rgb_{i}_visual.png"))

        semantic_input = image
        with torch.no_grad():
            if semantic_preprocessor != None:
                semantic_input = semantic_preprocessor(semantic_input)
            semantic_input = semantic_input.cuda()
            out = model(image)
            semantic_prediction = semantic_model(semantic_input)['out']
            semantic_out = semantic_prediction.softmax(dim=1)
            print(f"output size: {out.size()}")
            print(f"output size: {out.size(2), out.size(3)}")

            # Save depth image as numpy array
            depth_array = out.view(out.size(2), out.size(3)).data.cpu().numpy()
            depth_array = Image.fromarray(depth_array).resize((image.size(3), image.size(2)))
            depth_array = np.array(depth_array)
            np.save(os.path.join(output_path, f"depth_{i}.npy"), depth_array)
            print(f"Saved depth image with shape: {depth_array.shape}")

            # Save visualizable version of depth
            plt.imsave(os.path.join(output_path, f"depth_{i}_visual.png"), depth_array, cmap='viridis')

            # Optional: Save depth as TIFF for high precision visualization
            tifffile.imwrite(os.path.join(output_path, f"depth_{i}.tiff"), depth_array)

            # Save semantic images as numpy arrays
            for j in range(semantic_out.size(1)):
                semantic_array = semantic_out[0][j].data.cpu().numpy()
                semantic_array = Image.fromarray(semantic_array).resize((image.size(3), image.size(2)))
                semantic_array = np.array(semantic_array)
                np.save(os.path.join(output_path, f"semantic_{i}_{j if categories is None else categories[j]}.npy"), semantic_array)
                print(f"Saved semantic image {j} with shape: {semantic_array.shape}")

                # Save visualizable version of semantic
                plt.imsave(os.path.join(output_path, f"semantic_{i}_{j if categories is None else categories[j]}_visual.png"), semantic_array, cmap='jet')

        print(f"Processed and saved images for item {i}")
        
if __name__ == '__main__':
    main()

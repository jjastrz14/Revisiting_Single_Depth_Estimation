import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
from nyu_transform import *


class depthDataset(Dataset):
    
    """Face Landmarks dataset."""

    def __init__(self, npz_file, transform=None):
        self.frame = npz_file
        file = np.load(self.frame, mmap_mode='r')
        self.len = len(file['images'])
        file.close()
        self.transform = transform

    def __getitem__(self, idx):
        file = np.load(self.frame, mmap_mode='r')
        image = Image.fromarray(file['images'][idx])
        depth = Image.fromarray(file['depths'][idx])
        labels = Image.fromarray(file['labels'][idx])
        file.close()

        sample = {'image': image, 'depth': depth, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.len


def getTrainingData(batch_size=64):
    __imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([
            [-0.5675,  0.7192,  0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948,  0.4203],
        ])
    }
    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    transformed_training = depthDataset(npz_file='./data/nyu_depth_v2_labeled.npz',
                                        transform=transforms.Compose([
                                            Scale(240),
                                            RandomHorizontalFlip(),
                                            RandomRotate(5),
                                            CenterCrop([304, 228], [152, 114]),
                                            ToTensor(),
                                            Lighting(0.1, __imagenet_pca[
                                                'eigval'], __imagenet_pca['eigvec']),
                                            ColorJitter(
                                                brightness=0.4,
                                                contrast=0.4,
                                                saturation=0.4,
                                            ),
                                            Normalize(__imagenet_stats['mean'],
                                                      __imagenet_stats['std']),
                                            CombineWithLabel(),
                                        ]))
    
    len(transformed_training.__getitem__(0))

    dataloader_training = DataLoader(transformed_training, batch_size,
                                     shuffle=True, num_workers=4, pin_memory=False)

    return dataloader_training


def getTestingData(batch_size=64):

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    # scale = random.uniform(1, 1.5)
    transformed_testing = depthDataset(csv_file='./data/nyu2_test.csv',
                                       transform=transforms.Compose([
                                            Scale(240),
                                            CenterCrop([304, 228], [304, 228]),
                                            ToTensor(is_test=True),
                                            Normalize(__imagenet_stats['mean'],
                                                    __imagenet_stats['std']),
                                            CombineWithLabel(),
                                       ]))

    dataloader_testing = DataLoader(transformed_testing, batch_size,
                                    shuffle=False, num_workers=0, pin_memory=False)

    return dataloader_testing

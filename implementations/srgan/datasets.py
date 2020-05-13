import glob
import random
import os
import numpy as np

from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


def denormalize(tensors):
    """ Denormalizes image tensors using mean and std """
    for c in range(3):
        tensors[:, c].mul_(std[c]).add_(mean[c])
    return torch.clamp(tensors, 0, 255)


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, upscale_factor=4):
        hr_height, hr_width = hr_shape
        self.upscale_factor = upscale_factor
        self.hr_height = hr_height
        self.hr_width = hr_width
        # Transforms for low resolution images and high resolution images
        self.lr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height // upscale_factor, hr_height // upscale_factor), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.hr_transform = transforms.Compose(
            [
                transforms.Resize((hr_height, hr_height), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        self.gen_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.files = sorted(glob.glob(root + "/*.*"))
        self.is_both = len(self.files) and (('-lo.png' in self.files[0]) or ('-md.png' in self.files[0]) or ('-hi.png' in self.files[0]))

    def bothgetitem(self, index):
        hr_fn = self.files[index % len(self.files)].replace("-lo.png", "-hi.png").replace("-md.png", "-hi.png")
        lr_fn = hr_fn.replace("-hi.png", "-lo.png")
        hr_precrop = Image.open(hr_fn).convert('RGB')
        lr_precrop = Image.open(lr_fn).convert('RGB')
        crop_indices = RandomCrop.get_params(
            lr_precrop, output_size=(self.hr_width // self.upscale_factor, self.hr_height // self.upscale_factor))
        hr_crop_indices = tuple(ci * self.upscale_factor for ci in crop_indices)
        hr_cropped = TF.crop(hr_precrop, hr_crop_indices[0],hr_crop_indices[1],hr_crop_indices[2],hr_crop_indices[3])
        lr_cropped = TF.crop(lr_precrop, crop_indices[0],crop_indices[1],crop_indices[2],crop_indices[3])
        
        img_lr = self.gen_transform(lr_cropped)
        img_hr = self.gen_transform(hr_cropped)

        return {"lr": img_lr, "hr": img_hr}
        
        
    def __getitem__(self, index):
        if self.is_both:
            return self.bothgetitem(index)
        img = Image.open(self.files[index % len(self.files)])
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        if self.is_both:
            return len(self.files * 4)
        return len(self.files)

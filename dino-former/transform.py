import torchvision.transforms as transforms
from PIL import ImageFilter
import random

class GaussianBlur(object):
  def __init__(self,p):
    self.p=p
  def __call__(self,img):
    if random.random()<self.p:
      sigma=random.uniform(0.1,2.0)
      return img.filter(ImageFilter.GaussianBlur(radius=sigma))
    else:
       return img

class MulitCropTransform:
  def __init__(self,global_crops_scale,local_crops_scale,local_crops_number,global_crop_size=224,local_crops_size=96):
    self.global_transform1=transforms.Compose([
        transforms.RandomResizedCrop(global_crop_size,scale=global_crops_scale,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
    ])
    self.global_transform2=transforms.Compose([
        transforms.RandomResizedCrop(global_crop_size,scale=global_crops_scale,interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225))
    ])
    self.local_crops_number=local_crops_number
    self.local_transform=transforms.Compose([
        transforms.RandomResizedCrop(local_crops_size,scale=local_crops_scale,interpolation=transforms.InterpolationMode.BICUBIC), # Fixed typo
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),std=(0.229,0.224,0.225)),
    ])
  def __call__(self,img):
    crops=[]
    crops.append(self.global_transform1(img))
    crops.append(self.global_transform2(img))
    for _ in range(self.local_crops_number):
      crops.append(self.local_transform(img))
    return crops

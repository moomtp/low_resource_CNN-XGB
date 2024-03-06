import random
import os
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image



# csv label type
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, file_to_label_dict, transform=None, enable_gray_scale=False):
        self.img_dir = img_dir
        self.file_to_label_dict = file_to_label_dict

        self.transform = transform
        self.image_files = list(file_to_label_dict.keys())

        self.is_gray_scale = enable_gray_scale
    

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        if self.is_gray_scale:
            image = Image.open(img_path).convert('L')
        else :
            image = Image.open(img_path).convert('RGB')

        label = self.file_to_label_dict[self.image_files[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label


class CustomImageDatasetSubfolderType(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = []  # 存储图像文件路径和标签

        # 遍历目录，收集图像文件的路径和它们的标签
        for label_dir in os.listdir(img_dir):
            label_path = os.path.join(img_dir, label_dir)
            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    full_img_path = os.path.join(label_path, img_file)
                    if os.path.isfile(full_img_path):
                        self.images.append((os.path.join(full_img_path), label_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('L')  # 将图像转换为灰度
        # image = Image.open(img_path)  # 将图像转换为灰度
        if self.transform:
            image = self.transform(image)
        return image, label

import pandas as pd
import torch
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms, datasets
import json
from dataclasses import dataclass, field
from typing import List

# import helperFunction
import os
import sys
sys.path.append(os.path.abspath('..'))

from program.helperFunction.CustomImageDataset import CustomImageDatasetSubfolderType



@dataclass
class ChestCTDataProcessor:
    train_dataloader : DataLoader
    test_dataloader : DataLoader
    transform : transforms.Compose
    num_classes : int = 4
    current_dir : str = os.path.dirname(__file__)
    dataset_path : str = current_dir + "\\images"
    train_dataset_path : str = dataset_path + "\\train"
    test_dataset_path : str = dataset_path + "\\test"
    # feature_vector_file : str = current_dir + '\\img_feature_no_masked.csv'
    test_files : List[str] = field(default_factory=list)
    train_files : List[str] = field(default_factory=list)
    feature_vector_file = None


    def __init__(self, transform : transforms.Compose):
        self.transform = transform

        self.dataPreprocess()

    #  ======  interface function  ======
    def getDataloaders(self):
        return self.train_dataloader , self.test_dataloader
    def getDatasetFilenames(self):
        return None , None
    def getFeatureVectorFilename(self):
        return self.feature_vector_file
    def getNumClasses(self):
        return self.num_classes
    

    
    def dataPreprocess(self):
        # train_dataset = CustomImageDatasetSubfolderType(img_dir=self.train_dataset_path, transform=self.transform)
        train_dataset = datasets.ImageFolder(self.train_dataset_path, transform=self.transform)
        self.train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_dataset = datasets.ImageFolder(self.test_dataset_path, transform=self.transform)
        self.test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

        self.num_classes = len(test_dataset.classes)

        self.printDatasetInfo(test_dataset)



    def printDatasetInfo(self, dataset):
        # 查看类别
        print("Classes:", dataset.classes)

        # 查看类别到索引的映射
        print("Class to index:", dataset.class_to_idx)

        # 查看前几个样本
        print("Samples:", dataset.samples[:5])

        # 加载数据集中的第一个图像及其标签
        image, label = dataset[0]
        print("First image size:", image.size())
        print("First image label:", label)



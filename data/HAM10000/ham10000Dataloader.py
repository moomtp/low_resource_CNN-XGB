import pandas as pd
import torch
from torch.utils.data import DataLoader , Dataset
from torchvision import transforms
import json
from dataclasses import dataclass, field
from typing import List
import matplotlib.pyplot as plt
import random

# import helperFunction
import os
import sys
sys.path.append(os.path.abspath('..'))

from program.helperFunction.CustomImageDataset import CustomImageDataset

current_dir = os.path.dirname(__file__)

dataset_path = current_dir + "\images"
groundtruth_file = current_dir + '\GroundTruth.csv'

@dataclass
class HAM10000DataProcessor:
    train_dataloader : DataLoader
    test_dataloader : DataLoader
    transform : transforms.Compose
    current_dir : str = os.path.dirname(__file__)
    dataset_path : str = current_dir + "\images"
    groundtruth_file : str = current_dir + '\GroundTruth.csv'
    feature_vector_file : str = current_dir + '\img_feature_no_masked.csv'
    test_files : List[str] = field(default_factory=list)
    train_files : List[str] = field(default_factory=list)
    num_classes : int = 7

    def __init__(self, transform : transforms.Compose):
        self.transform = transform

        self.dataPreprocess()

    #  ======  interface function  ======
    def getDataloaders(self):
        return self.train_dataloader , self.test_dataloader
    def getDatasetFilenames(self):
        return self.train_files , self.test_files
    def getFeatureVectorFilename(self):
        return self.feature_vector_file
    def plotNumsSampleImg(self):
        # print(len(next(iter(self.train_dataloader))))
        self.show_random_image(self.train_dataloader)
        self.show_random_image(self.test_dataloader)
    def getNumClasses(self):
        return self.num_classes
    


    def dataPreprocess(self):

        # -----  定義 {檔案名 : label} 的dictionary  -----
        # 读取 CSV 文件
        groundtruth_data = pd.read_csv(groundtruth_file)

        def find_indices_of_ones(row):
            # 尋找前六個元素中 1 的位置
            return [(i-1) for i, x in enumerate(row[:8]) if x == 1]

        # 將函數應用於每行並創建新列 'label'
        groundtruth_data['label'] = groundtruth_data.apply(find_indices_of_ones, axis=1)

        # groundtruth_data.head()

        # find ele == 1
        filename_to_label_dict = groundtruth_data.set_index('image')['label'].to_dict()

        # {'1234' : [2] ,'1235' : [3] } -> {'1234' : 2 ,'1235' : 3 }
        filename_to_label_dict =  {key: value[0] if value else None for key, value in filename_to_label_dict.items()}

        # {'1234' : 2 ,'1235' : 3 } -> {'1234.jpg' : 2 ,'1235.jpg' : 3 }
        filename_to_label_dict =  {key + ".jpg": value for key, value in filename_to_label_dict.items()}

        # check if value is out of range
        null_keys = [key for key, value in filename_to_label_dict.items() if value is None]
        all_values_in_range = all(0 <= value <= 6 for value in filename_to_label_dict.values())
        assert all_values_in_range == True, "有些值不在範圍內"

          
        # -----  讀取影像資料，並使用filename_to_label_dict讓影像資料對應到label  -----
        # 讀取定義好的dataset filenames
        with open(current_dir + "\\test_files_list.json", 'r') as f:
            self.test_files = json.load(f)
        with open(current_dir + "\\train_files_list.json", 'r') as f:
            self.train_files = json.load(f)


        # 加載數據
        train_dataset = CustomImageDataset(img_dir=dataset_path,file_to_label_dict={file: filename_to_label_dict[file] for file in self.train_files}, transform=self.transform)
        self.train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        test_dataset = CustomImageDataset(img_dir=dataset_path,file_to_label_dict={file: filename_to_label_dict[file] for file in self.test_files}, transform=self.transform)
        self.test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    def show_random_image(self, dataloader):
        # 从dataloader中迭代获取一个批次的数据
        images, labels = next(iter(dataloader))
        
        # 随机选择一个图像
        idx = random.randint(0, len(images) - 1)
        image = images[idx]
        
        # 将图像的维度从(C, H, W)变换为(H, W, C)以适配matplotlib的显示要求
        # 注意：这个转换适用于单通道（例如灰度图像）和三通道（例如RGB图像）的情况
        # 如果图像是归一化的，可能还需要逆归一化步骤
        image = image.permute(1, 2, 0)
        
        # 使用matplotlib显示图像
        plt.imshow(image)
        plt.title(f'Label: {labels[idx]}')
        plt.show()

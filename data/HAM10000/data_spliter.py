import os
import shutil
import random

# 示例路径，请替换为实际路径
source_directory = 'images' 
destination_directory1 = './train_img_data' 
destination_directory2 = './test_img_data' 

random.seed(42)

# 确保目标目录存在
if not os.path.exists(destination_directory1):
    os.makedirs(destination_directory1)
if not os.path.exists(destination_directory2):
    os.makedirs(destination_directory2)

# 获取源目录中所有文件的列表
file_list = os.listdir(source_directory)

# 打乱文件列表
random.shuffle(file_list)

# 计算分割点
split_index = int(0.8 * len(file_list))

# 复制文件
for i, file_name in enumerate(file_list):
    source_file_path = os.path.join(source_directory, file_name)
    if i < split_index:
        # 前 80% 的文件复制到第一个目标目录
        shutil.copy(source_file_path, destination_directory1)
    else:
        # 剩余 20% 的文件复制到第二个目标目录
        shutil.copy(source_file_path, destination_directory2)

print("文件复制完成。")

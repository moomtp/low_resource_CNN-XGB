import glob

# 定义要查找的文件夹路径
folder_path = "./"

# 定义文件名的模式
name_pattern = "*_{}.csv".format('resnet18')

# 使用 glob 模块查找符合模式的文件
matching_files = glob.glob(folder_path + "/" + name_pattern)

# 输出匹配到的文件列表
print("匹配到的文件：", matching_files)
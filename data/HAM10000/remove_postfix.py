import os

folder_path = 'masks'
old_extension = '_segmentation.png'  # 替换为你要替换的旧后缀
new_extension = '.png'  # 替换为新后缀

# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(old_extension):
            old_file_path = os.path.join(root, file)
            new_file_path = old_file_path[:-len(old_extension)] + new_extension
            os.rename(old_file_path, new_file_path)
            print(f"已将文件重命名为: {new_file_path}")
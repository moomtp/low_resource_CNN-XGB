import os
import shutil

def copy_directory_structure(src_dir, dest_dir):
    """
    复制目录结构及文件。
    
    :param src_dir: 原始目录路径
    :param dest_dir: 目标目录路径
    """
    for root, dirs, files in os.walk(src_dir):
        # 计算当前目录相对于源目录的相对路径
        rel_path = os.path.relpath(root, src_dir)
        # 创建目标目录中对应的目录结构
        dest_path = os.path.join(dest_dir, rel_path)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        # 复制每个文件到新的目录
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(dest_path, file)
            shutil.copy2(src_file_path, dest_file_path)
            print(f"Copied {src_file_path} to {dest_file_path}")

# 示例用法
src_directory = './images'  # 替换为原始目录的路径
dest_directory = './copy_images'  # 替换为目标目录的路径

# 执行复制操作
copy_directory_structure(src_directory, dest_directory)
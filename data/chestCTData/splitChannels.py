import os
from PIL import Image

def save_channel_images_separately(directory, output_base_directory):
    # 定义通道名称，将用于创建子目录
    channels = ['R', 'G', 'B', 'A']

    # 为每个通道创建一个输出目录（如果它们还不存在）
    output_directories = {}
    for channel in channels:
        output_dir = os.path.join(output_base_directory, channel)
        output_directories[channel] = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # 遍历指定目录及其子目录中的所有图像文件
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否为支持的图像格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                with Image.open(image_path) as img:
                    # 检查图像是否为RGBA模式
                    if img.mode == 'RGBA':
                        # 分离RGBA通道
                        channels_images = img.split()
                        # 保存每个通道的图像到对应的子目录
                        for i, channel in enumerate(channels):
                            output_path = os.path.join(output_directories[channel], f"{os.path.splitext(file)[0]}_{channel}.png")
                            channels_images[i].save(output_path)
                            print(f"Saved {channel} channel image to {output_path}")

# 指定要遍历的目录路径
input_directory = 'images'
# 指定保存通道图像的基础目录路径
output_base_directory = 'channel_images'

# 调用函数
save_channel_images_separately(input_directory, output_base_directory)
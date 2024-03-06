import os
from PIL import Image

def overwrite_images_with_red_channel(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否为支持的图像格式
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                image_path = os.path.join(root, file)
                try:
                    with Image.open(image_path) as img:
                        # 检查图像是否为RGBA或RGB模式
                        if img.mode in ['RGBA', 'RGB']:
                            # 分离R通道
                            r, g, b = img.split()[:3]  # 仅取前三个通道以防是RGBA
                            # 将R通道转换为灰度图像
                            r.save(image_path)  # 直接覆盖原始图像
                            print(f"Overwritten {image_path} with its R channel")
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

# 指定要处理的目录路径
directory = './images'

# 调用函数
overwrite_images_with_red_channel(directory)
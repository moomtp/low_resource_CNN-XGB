import os
from PIL import Image

def remove_alpha_channel(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'tiff', 'bmp', 'webp')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        # 檢查影像是否包含 Alpha 通道
                        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                            # 去除 Alpha 通道
                            alpha_removed = img.convert('RGB')
                            # 保存修改後的影像，覆蓋原始檔案或保存到新位置
                            alpha_removed.save(file_path)
                            print(f'Alpha channel removed from {file_path}')
                except Exception as e:
                    print(f'Error processing file {file_path}: {e}')

# 設定你要處理的資料夾路徑
directory = './images'
remove_alpha_channel(directory)
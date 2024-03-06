import cv2
import os

# 指定影像和掩模文件夹路径
image_folder = 'images'
mask_folder = 'masks'
output_folder = 'masked_images'  # 保存提取的图像的文件夹

# 遍历影像文件夹和掩模文件夹
for image_filename in os.listdir(image_folder):
    if image_filename.endswith('.jpg'):  # 根据你的文件类型修改后缀
        image_path = os.path.join(image_folder, image_filename)
        mask_filename = image_filename.replace('.jpg', '.png')  # 假设掩模文件名与影像文件名相关
        mask_path = os.path.join(mask_folder, mask_filename)

        if os.path.exists(mask_path):
            # 读取影像和掩模
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # 如果掩模是灰度图像

            # 使用掩模提取需要的部分
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # 保存提取的图像
            output_filename = image_filename.replace('.jpg', '.jpg')
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, masked_image)

            print(f"已保存提取的图像: {output_path}")
        else:
            print(f"找不到对应的掩模文件: {mask_filename}")

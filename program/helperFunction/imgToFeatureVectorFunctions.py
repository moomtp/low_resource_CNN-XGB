from skimage import feature , io
from time import time
import csv
import cv2
import os
import numpy as np
import logging
import timeit

#######  feature computation function  ######

def computeLbpHistogram(image:np.ndarray, isRGB:bool = False):
    # 计算LBP特征
    img = image
    if isRGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(img, 8, 1, method='uniform')
    lbp = lbp.astype(np.uint8)

    # 计算LBP直方图
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 60), range=(0, 59))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # 归一化直方图

    return hist[:10]

def computeHuMoments_V0(image:np.ndarray, isRGB:bool = False):
    # 将图像转换为灰度图
    img = image
    if isRGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算Hu不变矩
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)

    # 将Hu不变矩转换为对数形式
    print(hu_moments)
    hu_moments = -1 * np.sign(hu_moments)

    nan_indices = np.where(np.isnan(hu_moments))
    if (len(nan_indices) > 1):
        print(f'Indices of NaN values: {list(zip(nan_indices[0]))}')
        # print("NaN ele found error!!")
        raise ValueError

    hu_moments = np.log10(np.abs(hu_moments))
    

    hu_moments = abs(hu_moments) / 30
    

    hu_moments = [item for subarr in hu_moments for item in subarr]
    
    hu_moments = np.array(hu_moments)

    return hu_moments

def computeHuMoments(image: np.ndarray, isRGB: bool = False):
    # 将图像转换为灰度图
    img = image
    if isRGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 计算Hu不变矩
    moments = cv2.moments(img)
    hu_moments = cv2.HuMoments(moments)

    # 对Hu不变矩的每个元素应用对数转换
    for i in range(0, 7):
        # 避免对 0 或非常接近 0 的值取对数
        hu_moments[i] = hu_moments[i] if abs(hu_moments[i]) > 1e-10 else 1e-10
        hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))

    # 计算均值和标准差
    mean = np.mean(hu_moments)
    std = np.std(hu_moments)

    # 进行 z-score 标准化
    hu_moments = (hu_moments- mean) / std
    # 将Hu不变矩展平成一维数组
    hu_moments = hu_moments.flatten()


    nan_indices = np.where(np.isnan(hu_moments))
    # print(len(nan_indices))
    if (len(nan_indices) != 1):
        print(hu_moments)
        print(f'Indices of NaN values: {list(zip(nan_indices[0]))}')
        # print("NaN ele found error!!")
        raise ValueError

    return np.array(hu_moments)


def computeColorIndex(image:np.ndarray, num_bins=8, isRGB:bool=False):
    # 转换图像为Lab颜色空间
    if isRGB == False:
        return np.array([])
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # 计算图像颜色直方图
    hist_L = cv2.calcHist([lab_image], [0], None, [num_bins], [0, 256])
    hist_A = cv2.calcHist([lab_image], [1], None, [num_bins], [0, 256])
    hist_B = cv2.calcHist([lab_image], [2], None, [num_bins], [0, 256])

    # 归一化直方图
    hist_L = cv2.normalize(hist_L, hist_L).flatten()
    hist_A = cv2.normalize(hist_A, hist_A).flatten()
    hist_B = cv2.normalize(hist_B, hist_B).flatten()

    # print("hsv vector type:")
    # print(hist_L)

    # 合并直方图为单个特征向量
    color_index = np.concatenate((hist_L, hist_A, hist_B))

    return color_index

#  ===============    describe is from kaggle compatition   ================

# def HSVdescribe(image, bins=[8,12,8]):
#     # Create center mask
#     (h, w) = image.shape[:2]
#     (cX, cY) = (int(w * 0.5), int(h * 0.5))
#     (axesX, axesY) = (int(w * 0.5) // 2, int(h * 0.6) // 2)
#     mask = np.zeros(image.shape[:2], dtype = "uint8")
#     cv2.ellipse(mask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
    
#     # Get 3D HSV color histogram
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([image], [0, 1, 2], mask, bins, [0, 256, 0, 256, 0, 256])
#     hist = cv2.normalize(hist,hist).flatten()
#     return hist

# def LBPdescribe(image, numPoints=126, radius=3, eps=1e-7):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
#     (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, numPoints + 3), range=(0, numPoints + 2))
#     hist = hist.astype("float")
#     hist /= (hist.sum() + eps)
#     return hist

#  combination of three feature vector
def calFeatureVector(image):
    vec1 = computeColorIndex(image)   
    vec2 = computeHuMoments(image)   
    vec3 = computeLbpHistogram(image)   
    # print(vec1)
    # print(vec2)
    # print(vec3)

    # print(type(vec1))
    # print(type(vec2))
    # print(type(vec3))

    vec = np.concatenate((vec1, vec2 , vec3))
    return vec.tolist()



if __name__ == "__main__":
    start_time = time()
    input_folder = "test/images/"
    csvfile_name = "./img_feature.csv"


    res = []

    # open every img
    for filename in os.listdir(input_folder):

        image_path  = os.path.join(input_folder, filename)
        print(image_path)
        image = cv2.imread(image_path)
        vec = calFeatureVector(image)
        vec = [filename] + vec
        res.append(vec)

    # print(res)
    with open(csvfile_name, 'w', newline='', encoding='utf-8') as csvfile:
        # try:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(res)
        # except Exception as e:
        #     logging.error("csv error : %s" , e)
        #     print('csv encoding error')
    
    end_time = time()

    print("總耗時{} 秒".format(end_time - start_time))
from skimage import feature , io
from time import time
import csv
import cv2
import os
import numpy as np
import logging
import timeit
import program.helperFunction.imgToFeatureVector as imgToFeatureVector


if __name__ == "__main__":
    start_time = time()
    input_folder = "test/images/"
    csvfile_name = "./img_feature.csv"
    image_path = "test/images/ISIC_0024306.jpg"


    res = []

    # open every img

    image = cv2.imread(image_path)

    start_time = time()
    for i in range(1000):
        vec = imgToFeatureVector.compute_lbp_histogram(image)
    end_time = time()
    print("單張耗時{} ms".format((end_time - start_time)))
    
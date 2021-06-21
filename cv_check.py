import cv2
import numpy as np
import os
import re

img_name = r"0000a68812bc7e98c42888dfb1c07da0.jpg"
# img_name = "00ca272935530e43f8574b3ef967cddb.jpg"
base_path = os.getcwd()
path = os.path.join(base_path, 'train_images', img_name)
print(path)
# path = re.sub(r'\\', r'\\\\', path)
# print(path)
img = cv2.imread(path, cv2.IMREAD_COLOR)
# img = np.random.random((512, 512, 3))
# img = np.empty((512, 512, 3))
img = cv2.resize(img, (512, 512))
h, w, c = img.shape
norm = np.empty((h, w))
normimg = cv2.normalize(img,  None, 0, 255, cv2.NORM_MINMAX)
# cv2.imshow('Normalized Image', normimg)
# cv2.imshow('Normalized Image', img)
# cv2.waitKey(0)

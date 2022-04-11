# -*- coding: utf-8 -*-
# !/usr/bin/env python
# 余弦相似度计算
from PIL import Image
from numpy import average, dot, linalg
# 对图片进行统一化处理
def get_thum(image, size=(64, 64), greyscale=False):
    # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
    image = image.resize(size, Image.ANTIALIAS)
    if greyscale:
        # 将图片转换为L模式，其为灰度图，其每个像素用8个bit表示
        image = image.convert('L')
    return image
# 计算图片的余弦距离
def image_similarity_vectors_via_numpy(image1, image2):
    image1 = get_thum(image1)
    image2 = get_thum(image2)
    images = [image1, image2]
    vectors = []
    norms = []
    for image in images:
        vector = []
        for pixel_tuple in image.getdata():
            vector.append(average(pixel_tuple))
        vectors.append(vector)
        # linalg=linear（线性）+algebra（代数），norm则表示范数
        # 求图片的范数
        norms.append(linalg.norm(vector, 2))
    a, b = vectors
    a_norm, b_norm = norms
    # dot返回的是点积，对二维数组（矩阵）进行计算
    res = dot(a / a_norm, b / b_norm)
    return res

def cosin():
    root = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/compare/'
    image1 = Image.open(root + 'forest.tif')
    image2 = Image.open(root + 'water.tif')
    cosin = image_similarity_vectors_via_numpy(image1, image2)
    print('图片余弦相似度', cosin)


"""
        farmland forest water
farmland          0.956 0.943
forest                  0.935
water
"""

# 将图片转化为RGB
def make_regalur_image(img, size=(64, 64)):
    gray_image = img.resize(size).convert('RGB')
    return gray_image


# 计算直方图
def hist_similar(lh, rh):
    assert len(lh) == len(rh)
    hist = sum(1 - (0 if l == r else float(abs(l - r)) / max(l, r)) for l, r in zip(lh, rh)) / len(lh)
    return hist


# 计算相似度
def calc_similar(li, ri):
    calc_sim = hist_similar(li.histogram(), ri.histogram())
    return calc_sim


def hist():
    root = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/compare/'
    image1 = Image.open(root + 'farmland.tif')
    image2 = Image.open(root + 'forest.tif')
    image1 = make_regalur_image(image1)
    image2 = make_regalur_image(image2)
    print("图片间的相似度为", calc_similar(image1, image2))

"""
        farmland forest water
farmland          0.587 0.388
forest                  0.341
water
"""

# -*- coding: utf-8 -*-
from skimage.metrics import structural_similarity
from imageio import imread
import numpy as np
import cv2
# 读取图片
root = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/compare/'
img1 = imread(root + 'forest.tif')
img2 = imread(root + 'water.tif')
img2 = np.resize(img2, (img1.shape[0], img1.shape[1], img1.shape[2]))
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
print(gray1.shape)
print(gray2.shape)
ssim =  structural_similarity(gray1, gray2)
print(ssim)
# 比较灰度图
"""
        farmland forest water
farmland          0.237 0.265
forest                  0.170
water
"""
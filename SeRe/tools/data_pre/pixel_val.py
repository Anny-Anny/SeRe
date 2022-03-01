import os

from PIL import Image
import numpy as np
import cv2

# “L”表示转灰度
# img_L = np.array(Image.open('test.png').convert("L"))
# img_RGB = np.array(Image.open('test.png').convert("RGB"))

# temp = {}
# for i in range(img_L.shape[0]):
#   for j in range(img_L.shape[1]):
#     if not temp.get(int(img_L[i][j])):
#       temp[int(img_L[i][j])] = list(img_RGB[i][j])
# print(temp)

Dataroot = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/potsdam/6_Labels_all_gray/'
a = []


def pixel_val(img_path):
    img_L = np.array((Image.open(Dataroot + img_path)))
    # 这里得到灰度像素值0对应(0,0,0),62对应(19,69,139)
    color_0_0_0 = np.where(img_L == 0)[0].shape[0]
    color_0_0_1 = np.where(img_L == 1)[0].shape[0]
    color_0_0_2 = np.where(img_L == 2)[0].shape[0]
    color_0_0_3 = np.where(img_L == 3)[0].shape[0]
    color_0_0_4 = np.where(img_L == 4)[0].shape[0]
    color_0_0_5 = np.where(img_L == 5)[0].shape[0]
    color_0_0_6 = np.where(img_L == 6)[0].shape[0]

    # if color_0_0_6 > 0:
    #     a.append('哭了')

    pixel_sum = img_L.shape[0] * img_L.shape[1]

    print("0_0_0 像素个数：{} 占比：%{}".format(color_0_0_0, color_0_0_0 / pixel_sum * 100))
    print("0_0_1 像素个数：{} 占比：%{}".format(color_0_0_1, color_0_0_1 / pixel_sum * 100))
    print("0_0_2 像素个数：{} 占比：%{}".format(color_0_0_2, color_0_0_2 / pixel_sum * 100))
    print("0_0_3 像素个数：{} 占比：%{}".format(color_0_0_3, color_0_0_3 / pixel_sum * 100))
    print("0_0_4 像素个数：{} 占比：%{}".format(color_0_0_4, color_0_0_4 / pixel_sum * 100))
    print("0_0_5 像素个数：{} 占比：%{}".format(color_0_0_5, color_0_0_5 / pixel_sum * 100))
    print("0_0_6 像素个数：{} 占比：%{}".format(color_0_0_6, color_0_0_6 / pixel_sum * 100))


if __name__ == "__main__":
    imgs = os.listdir(Dataroot)
    for ch in imgs:
        print(ch)
        pixel_val(ch)
    # print('a:')
    # print(a)

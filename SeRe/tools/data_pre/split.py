import os

import cv2
import numpy as np
from collections import namedtuple
import scipy.io
import imageio
import random
from PIL import Image, ImageFont, ImageDraw
import ade20k_labels

# DATAROOT = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/potsdam'
DATAROOT = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan'

def split_sigle_img(img_name, label_name, start_index):
    img1 = cv2.imread(img_name)  # 读取RGB原图像

    img2 = cv2.imread(label_name)  # 读取Labels图像

    # 因为6000/224 = 26，所以6000x6000的图像可以划分为26x26个224x224大小的图像
    for i in range(23):
        for j in range(23):
            img1_ = img1[256 * i: 256 * (i + 1), 256 * j: 256 * (j + 1), :]
            img2_ = img2[256 * i: 256 * (i + 1), 256 * j: 256 * (j + 1), 0]

            name = start_index + i * 23 + j
            # 让RGB图像和标签图像的文件名对应
            name = str(name)
            cv2.imwrite(DATAROOT + '/images/' + name + '.tif', img1_)  # 所有的RGB图像都放到jpg文件夹下
            cv2.imwrite(DATAROOT + '/annotations/' + name + '.tif', img2_)  # 所有的标签图像都放到png文件夹下


def split_potsdam_dataset(dataset_img_path, dataset_label_path):
    name_list = os.listdir(dataset_label_path)  # 包含 .tif 后缀
    for i, ch in enumerate(name_list):
        name = ch.split('.')[0][:-6]
        img_name = dataset_img_path + '/' + name + '_RGB.tif'
        label_name = dataset_label_path + '/' + name + '_label.tif'
        start_index = 529 * i
        split_sigle_img(img_name, label_name, start_index)
        print("第" + str(i) + "张裁剪完成！")

def gen_txt():
    '''
    @TODO:添加按比例划分的功能
    '''
    path = ''
    # path = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan/train.txt'
    # path = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/potsdam/images'

    names = []
    if path.endswith('.txt'):
        with open(path, 'r') as f:
            name_list = f.readlines()
            for line in name_list:
                line = line.strip()
                names.append(line)
    # 40% 8040 40% 8040 20% 4022 一共10202
    if not path.endswith('.txt'):
        name_list = os.listdir(path)
        for ch in name_list:
            names.append(int((ch.split('.')[0])))
    # 随机打乱顺序
    random.shuffle(names)
    # with open(DATAROOT + '/train.txt', 'w') as f:
    #     for name in names[:16080]:
    #         f.write(str(name) + '\n')
    # with open(DATAROOT + '/val.txt', 'w') as f:
    #     for name in names[16080:]:
    #         f.write(str(name) + '\n')
    # with open(DATAROOT + '/ori.txt', 'w') as f:
    #     for name in names[:8040]:
    #         f.write(str(name) + '\n')
    # with open(DATAROOT + '/new.txt', 'w') as f:
    #     for name in names[8040:16080]:
    #         f.write(str(name) + '\n')

    with open(DATAROOT + '/ori.txt', 'w') as f:
        for name in names[:7946]:
            f.write(str(name) + '\n')
    with open(DATAROOT + '/new.txt', 'w') as f:
        for name in names[7946:]:
            f.write(str(name) + '\n')


def cal_percent(image):
    """
    Class Name,B,G,R
    0,0,0,0
    低矮植被,0,255,255
    其它,255,255,255
    树木,0,255,0
    建筑物,255,0,0
    不透水面,0,0,255
    汽车,255,255,0

    surfaces (RGB: 255, 255, 255)
    Building (RGB: 0, 0, 255)
    low vegetation (RGB: 0, 255, 255)
    Tree (RGB: 0, 255, 0)
    Car (RGB: 255, 255, 0)
    Clutter / background (RGB: 255, 0, 0)
    :return:
    """

    rgb = {}
    name = ade20k_labels.ade20k_id2label  # 这个是颜色列表，自行创建一个即可
    # 这个是颜色列表，自行创建一个即可
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if (tuple(image[row, col]) not in rgb) or (row == 0 and col == 0):
                rgb[tuple(image[row, col])] = [1, tuple(image[row, col])]
            else:
                rgb[tuple(image[row, col])][0] = rgb[tuple(image[row, col])][0] + 1
    # rgb{颜色:[数量,颜色]}
    num = 0
    # for k in rgb:
    #     rgb[k][0] = rgb[k][0] / 65536
    #     num = num + rgb[k][0]

    # print(rgb)
    # print(num)
    return rgb


def write_image(img, write_contents, text_size=10):
    '''在图片中写上文字'''

    # 设置字体(从windows/font中查找自己所需要的字体)
    fontpath = 'font/simsun.ttc'  # （宋体）
    font = ImageFont.truetype(fontpath, text_size)  # 设置字体为ImageDeaw.draw()服务
    if type(img) is np.ndarray:
        image = Image.fromarray(img)  # array转换成Image(反过来则使用np.array())
    else:
        image = img
    draw = ImageDraw.Draw(image)
    # 绘制文字信息
    draw.text((0, 0), write_contents, font=font, fill=(0, 0, 0))

    return img


def create_rectangle(shape, color, text, text_size=10):
    '''创建正方形'''

    rectangle = Image.new('RGB', shape, color)
    rectangle = write_image(rectangle, text, text_size)
    return rectangle


def legend(image, text_size=10):
    ''' 创建图例 '''

    # 计算图像的各颜色百分比
    shape = np.array(image).shape
    h = int(shape[1] / 10)
    print(shape)

    rgb, num = cal_percent(np.asarray(image))
    print(num)
    # img = Image.new('RGB', (shape[1], int(num / 10 + 1) * 60), (96, 96, 96))  # 第二个参数为（height，width）
    # img2 = Image.new('RGB', (shape[1], int(num / 10 + 1) * 60 + shape[0]), (255, 255, 255))
    # print(img2)

    # 绘制图例
    # for i, k in enumerate(rgb):
    #     if i % 10 == 0:
    #         count1 = 0
    #     else:
    #         count1 = count1 + 1
    #     text = str(k) + '(' + rgb[k][0] + ')'
    #     rectangle = create_rectangle((h - 20, 30), rgb[k][1], text, text_size)
    #     coordination = (count1 * h, 30 + int(i / 10) * 60)  # 第二个参数为（height，width）
    #     img.paste(rectangle, coordination)
    #
    # img2.paste(image, (0, 0))
    # img2.paste(img, (0, shape[0]))
    # return img2


def cal_dataset(dataset_path):
    img_list = os.listdir(dataset_path)
    label = {(255, 255, 255): 'surfaces',
             (0, 0, 255): 'building',
             (0, 255, 255): 'low vegetation',
             (0, 255, 0): 'Tree',
             (255, 255, 0): 'Car',
             (255, 0, 0): 'Clutter / background'
             }
    cal = {(255, 255, 255): 0,
           (0, 0, 255): 0,
           (0, 255, 255): 0,
           (0, 255, 0): 0,
           (255, 255, 0): 0,
           (255, 0, 0): 0
           }
    bad = []
    # 输入进行统计的数量范围
    a = img_list[:5]
    for i, ch in enumerate(a):
        image1 = cv2.imread(dataset_path + '/' + ch)
        rgb = cal_percent(np.asarray(image1))
        if len(rgb.keys()) > 6:
            print(ch)
            bad.append(ch)
        for k in cal:
            if k in rgb.keys():
                cal[k] += rgb[k][0]
        print('第' + str(i) + "张" + ch)
        # print(sum(cal.values()))
        s = 256 * 256 * len(a)
    for k in cal.keys():
        cal[k] = cal[k] / s
    for k in cal.keys():
        print("%s %.2f" % (label[k], cal[k]))
    print(sum(cal.values()))
    print("bad ones")
    print(bad)


if __name__ == '__main__':
    # split_potsdam_dataset('/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/potsdam/2_Ortho_RGB',
    #                       '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/potsdam/6_Labels_all_gray')
    gen_txt()
    # image1 = cv2.imread('/home/xjw/下载/G-RSIM-main/T-SS-GLCNet/png/55.tif')
    # cal_percent(np.asarray(image1))
    # 计算标注文件的目录下，所有类别的像素占比
    # dataset_path = '/media/xjw/data/seg_cl/myseg_cl/data_potsdam/annotations'
    # cal_dataset(dataset_path)

"""{(255, 255, 0): ['3.2564%', (255, 255, 0)], (0, 255, 0): ['0.8195%', (0, 255, 0)], (255, 255, 255): ['1.6645%', (255, 255, 255)], (255, 0, 0): ['1.0864%', (255, 0, 0)]}"""

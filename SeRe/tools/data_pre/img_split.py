# Python程序
import cv2
import numpy

for k in range(7, 14):
    img1 = cv2.imread('/user-data/GNN_RemoteSensor/2_Ortho_RGB/top_potsdam_7_' + str(k) + '_RGB.tif')  # 读取RGB原图像
    img2 = cv2.imread('/user-data/GNN_RemoteSensor/5_Labels_all/top_potsdam_7_' + str(k) + '_label.tif')  # 读取Labels图像
    # 因为数据集中图片命名不规律，所以需要一批一批的分割
    # cv2.imread函数会把图片读取为（B，G，R）顺序，一定要注意！！！
    # 因为6000/256 = 23，所以6000x6000的图像可以划分为23x23=529个256x256大小的图像
    for i in range(10):
        for j in range(10):
            img1_ = img1[256 * i: 256 * (i + 1), 256 * j: 256 * (j + 1), :]
            img2_ = img2[256 * i: 256 * (i + 1), 256 * j: 256 * (j + 1), :]
            # 注意下面name的命名，2400 + k * 100需要一批一批的调整，自己看到数据集中的图片命名就能知道什么意思了
            name = i * 10 + j + 2400 + k * 100
            # 让RGB图像和标签图像的文件名对应
            name = str(name)
            cv2.imwrite('./datasets/images/' + name + '.jpg', img1_)  # 所有的RGB图像都放到jpg文件夹下
            cv2.imwrite('./datasets/labels/' + name + '.png', img2_)  # 所有的标签图像都放到png文件夹下

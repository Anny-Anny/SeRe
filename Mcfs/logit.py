import cv2
import numpy as np
import torch
from skimage import io
from visdom import Visdom
import time
import matplotlib.pyplot as plt



def output2tif(output,index:int,tif_path):
    # 将某类的概率输出转成tif进行保存
    # 承接自decoder的output = F.softmax(seg_logit, dim=1)
    # 一共有9类 需要输出索引值为 index 的类别置信度：如耕地 1
    output = output[0,index,:,:]
    output = output.cpu().numpy()
    cv2.imwrite(tif_path,output)


def show_tif(tif_path):
    # 可视化概率输出tif图片
    tif = io.imread(tif_path)  # cv2不能读取32 bit 的图片。所以使用skimage去读取
    # tif = torch.from_numpy(tif)
    vis = Visdom()
    vis.heatmap(tif) # 绘制热力图
    # vis.image(tif) # 直接以黑白灰度图的形式展示

def minus(tif1,tif2,gt,target_class):
    # 计算耕地的耕地概率减少量
    # tif1是模型a的推理概率，tif2是模型b的推理概率
    gt = io.imread(gt)
    mask = (gt == target_class) # mask保留耕地,~mask表示取反
    tif1 = io.imread(tif1)
    tif2 = io.imread(tif2)
    delta0 = tif2 - tif1 # 全部像素的概率变化情况
    delta = (tif2 - tif1)*mask # gt为耕地的耕地概率变化
    delta1 = (tif2 - tif1) * (~mask) # gt为非耕地的耕地概率变化)
    # vis = Visdom()
    # vis.heatmap(delta0)
    # vis.heatmap(delta)
    # vis.heatmap(delta1)
    a = np.sum(gt == True)
    b = np.sum(gt !=True)
    # 计算耕地
    a_ho = np.sum(delta == 0)
    a_up = np.sum(delta > 0)
    a_up_v = sum(delta[delta>0])/a_up
    a_down = np.sum(delta < 0)
    a_down_v = sum(delta[delta < 0]) / a_down
    # 计算非耕地
    b_ho = np.sum(delta1 == 0)
    b_up =   np.sum(delta1 > 0)
    b_up_v = sum(delta1[delta1 > 0]) / b_up
    b_down = np.sum(delta1 < 0)
    b_dwon_v = sum(delta1[delta1 < 0]) / b_down
    num = gt.shape[0] * gt.shape[1]
    print("耕地的耕地概率不变的像素有%d个，占比%.2f%%" % (a_ho, a_ho/num))
    print("耕地的耕地概率下降的像素有%d个，占比%.2f%%，平均%.3f" % (a_down, a_down / num, a_down_v))
    print("耕地的耕地概率上升的像素有%d个，占比%.2f%%, 平均+%.3f" % (a_up, a_up/num,a_up_v))
    print("非耕地的耕地概率不变的像素有%d个，占比%.2f%%" % (b_ho, b_ho / num))
    print("非耕地的耕地概率下降的像素有%d个，占比%.2f%%，平均%.3f" % (b_down, b_down / num,b_dwon_v))
    print("非耕地的耕地概率上升的像素有%d个，占比%.2f%%, 平均+%.3f" % (b_up, b_up / num,b_up_v))
    # cv2.imwrite('delta',delta)
# 把0算作上升
# 耕地的耕地概率上升的像素有47314个，占比0.72%
# 耕地的耕地概率下降的像素有18222个，占比0.28%
# 非耕地的耕地概率上升的像素有45601个，占比0.70%
# 非耕地的耕地概率下降的像素有19935个，占比0.30%
def round_show(tif1,tif2):
    tif1 = io.imread(tif1)
    tif2 = io.imread(tif2)
    vis = Visdom()
    for i in range(100):
        if i % 2 == 0:
            vis.heatmap(tif1,win = 'a')
        else:
            vis.heatmap(tif2, win='a')
        time.sleep(0.5)


# show_tif('/home/xjw/Downloads/code/mmsegmentation-0.21.0/Mcfs/farm_logit1.tif')
# show_tif('/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/val/annotations/xiangtan__19__204.tif')
# round_show('/home/xjw/Downloads/code/mmsegmentation-0.21.0/Mcfs/farm_logit.tif',
#            '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/val/annotations/xiangtan__19__204.tif')
# vis = Visdom()
# vis.text("Hello World!")
minus('/home/xjw/Downloads/code/mmsegmentation-0.21.0/Mcfs/farm_logit.tif',
      '/home/xjw/Downloads/code/mmsegmentation-0.21.0/Mcfs/farm_logit1.tif',
    '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/val/annotations/xiangtan__19__204.tif',
      1)
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmcv
from mmseg.core.evaluation import eval_hooks, eval_metrics,intersect_and_union, mean_iou
import cv2
import numpy as np
from PIL import Image
import os
def test_sig_img(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
                 checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori_work_dirs/myconfig/iter_60000.pth',
                 img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__11__50.tif',
                 show = False,
                 save = False):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, img)
    if show:
        show_result_pyplot(model, img, result, opacity=0.5) # only to show the result ,no save
    # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    if save:
        model.show_result(img, result, out_file='./result.jpg', opacity=0.5)
    return result

def vis_img_gr(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
                 checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori_work_dirs/myconfig/iter_60000.pth',
                 img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__11__50.tif',
                 gr = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/annotations/xiangtan__11__50.tif',
                 show = False,
                 save = True):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    # result = inference_segmentor(model, img)
    img = mmcv.imread(img)
    result = mmcv.imread(gr)
    if show:
        show_result_pyplot(model, img, result, opacity=0.5) # only to show the result ,no save
    # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    if save:
        model.show_result(img, result, out_file='./result1.jpg', opacity=0.5)
    return result

def cal_sig_aAcc(config_file,checkpoint_file,img,gt_file):
    # model = init_segmentor(config_file, checkpoint_file, device='cuda:0') # 这里应该可以注释掉，不需要加载model
    img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__11__50.tif'  # or img = mmcv.imread(img), which will only load it once
    # gt_file = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/annotations/xiangtan__11__50.tif'
    gt = cv2.imread(gt_file, -1)
    result = test_sig_img(config_file,checkpoint_file,img)
    eval_res = eval_metrics(result, gt, 9, ignore_index= 10, ) # 为什么一定要输入一个要被忽略的类别呢,感觉没啥影响是因为之前写的0类根本是NAN
    # gt = gt.tolist()
    # IoU = intersect_and_union(np.array(result)[0], gt, 9, ignore_index= 10, ) #不知道转成了多维数组就自动多了一个维度，所以取了[0]
    # mIoU  = mean_iou(np.array(result)[0], gt, 9, ignore_index= 10, )
    # print("eval_res is : ", eval_res)
    # print("IoU is : ", IoU)
    # print("mIoU is : ", mIoU)
    aAcc = eval_res['aAcc'].tolist()
    return aAcc

if __name__ == '__main__':
    # aAcc = cal_sig_aAcc(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
    #         checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori_work_dirs/myconfig/iter_60000.pth',
    #         img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__11__50.tif',
    #         gt_file='/media/xjw/data/seg_cl/myseg_cl/data/new_data/annotations/xiangtan__11__50.tif')
    # print(aAcc)


    # 经过我严密的推理，eval_metrics和mIoU计算的是同一个东西（nonono）
    # 测试单张图片并且可视化
    test_sig_img(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
                 checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori1_work_dirs/myconfig/iter_80000.pth',
                 img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__10__205.tif',
                 show=False,
                 save=True)
    # vis_img_gr()
        
# 11_50,10_205
#{'aAcc': array(95.36328125), 'IoU': array([       nan, 0.83421438,        nan, 0.19274611, 0.        ,
#        0.62170841,        nan, 0.09732143,        nan]), 'Acc': array([         nan, 121.27777778,          nan,  19.92857143,
#                 nan,  97.68807339,          nan,   9.90909091,
#                 nan])}



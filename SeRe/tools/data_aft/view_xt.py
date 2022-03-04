import shutil

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmcv
from mmseg.core.evaluation import eval_hooks, eval_metrics,intersect_and_union, mean_iou
import cv2
import numpy as np
from PIL import Image
import os
def test_sig_img(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig1.py',
                 checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori_work_dirs/myconfig/iter_60000.pth',
                 img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__11__50.tif',
                 show = False,
                 save = False,
                 save_name= 'result'):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, img)
    if show:
        show_result_pyplot(model, img, result, opacity=0.7) # only to show the result ,no save
    # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    if save:
        model.show_result(img, result, out_file= save_name + '.jpg', opacity=0.7)
    return result

def label(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
                 checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori_work_dirs/myconfig/iter_60000.pth',
                 img = '/media/xjw/data/seg_cl/myseg_cl/data/xiangtan_cjdata/images/validation/xiangtan__7__66.tif',
                 pngfile = '/media/xjw/data/seg_cl/myseg_cl/data/xiangtan_cjdata/annotations/validation/xiangtan__7__66.tif',
                 show = False,
                 save = False,
                 save_name= 'result'):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, img)
    png = [cv2.imread(pngfile,-1)]
    if show:
        show_result_pyplot(model, img, result, opacity=0.7) # only to show the result ,no save
    # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    if save:
        model.show_result(img, result, out_file= save_name + '.jpg', opacity=0.7)

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

def copy():
    file = {'__7__66': 'bg',
            '__105__190': 'farmland',
            '__68__80': 'town',
            '__79__151': 'village',
            '__55__59': 'water',
            '__68__192': 'forest',
            '__37__179': 'grass',
            '__31__145': 'road'}
    for a, b in file.items():
        img = '/media/xjw/data/seg_cl/myseg_cl/data/xiangtan_cjdata/images/validation/xiangtan' + a + '.tif'
        current_path = os.getcwd()
        print(current_path)
        path = current_path + '/' + b
        shutil.copy(img,path)
def save_vis():
    # 测试单张图片并且可视化
    file = {'__7__66': 'bg',
            '__105__190': 'farmland',
            '__68__80': 'town',
            '__79__151': 'village',
            '__55__59': 'water',
            '__68__192': 'forest',
            '__37__179': 'grass',
            '__31__145': 'road'}
    model = ['/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/ori1_work_dirs/myconfig/iter_80000.pth',
            '/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/_r_20%_work_dirs/iter_16000.pth',
            '/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/_union_work_dirs/iter_160000.pth',
            '/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/_rp_m_uni_25%_work_dirs/epoch_20.pth']
    dirname = ['static','finetune','union','sere','label']
    for a,b in file.items():
        img = '/media/xjw/data/seg_cl/myseg_cl/data/xiangtan_cjdata/images/validation/xiangtan' + a + '.tif'
        png = '/media/xjw/data/seg_cl/myseg_cl/data/xiangtan_cjdata/annotations/validation/xiangtan' + a + '.tif'
        current_path = os.getcwd()
        print(current_path)
        path = current_path + '/' + b
        if not os.path.exists(path):
            os.mkdir(path)
        for i, mo in enumerate(model):
            test_sig_img(config_file ='/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
                             checkpoint_file =mo,
                             img = img,
                             show=False,
                             save=True,
                             save_name=path + '/' + dirname[i])
            # label(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
            #      checkpoint_file = mo,
            #      img = img,
            #      pngfile = png,
            #      show = False,
            #      save = True,
            #      save_name= path + '/' + dirname[-1])
    # __7__66 bg
    # __105__190 farmland
    # __68__80 town
    # __79__151  village
    # __55__59 water
    # __68__192 forest
    # __37__179 grass
    # __31__145 road
    # vis_img_gr()
# save_vis()
# label()
copy()

        
""" 
测试各种模型的图片编号
11_50（最初的,各种类型都有，有5类）,10_205(耕地居多)，10__208(水体居多，精度提升明显),31__41(道路居多，精度提升明显)

模型路径
/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/ori1_work_dirs/myconfig/iter_80000.pth
/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/_r_20%_work_dirs/iter_16000.pth
/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/_union_work_dirs/iter_160000.pth
/media/xjw/data/seg_cl/mmsegmentation-master/.xiangtan_work_dirs/_rp_m_uni_25%_work_dirs/epoch_20.pth
"""
#{'aAcc': array(95.36328125), 'IoU': array([       nan, 0.83421438,        nan, 0.19274611, 0.        ,
#        0.62170841,        nan, 0.09732143,        nan]), 'Acc': array([         nan, 121.27777778,          nan,  19.92857143,
#                 nan,  97.68807339,          nan,   9.90909091,
#                 nan])}



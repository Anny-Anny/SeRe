import shutil

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmcv
from mmseg.core.evaluation import eval_hooks, eval_metrics,intersect_and_union, mean_iou
import cv2
import numpy as np
from PIL import Image
import os
def test_sig_img(config_file = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/Mcfs/config/myconfig_xt1.py',
                 checkpoint_file = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/workdir/mcfs/forest+farmland/epoch_100.pth',
                 img = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/val/images/xiangtan__19__204.tif',
                 show = True,
                 save = False,
                 save_name= 'result'):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, img)
    if show:
        show_result_pyplot(model, img, result, opacity=0.7) # only to show the result ,no save

    # you can change the opacity of the painted segmentation map in (0, 1].
    if save:
        model.show_result(img, result, out_file= save_name + '.jpg', opacity=0.7)
    return result

def label(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
                 checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori_work_dirs/myconfig/iter_60000.pth',
                 img = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/photo/images/xiangtan__7__164.tif',
                 pngfile = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan_oo/photo/annotaions/xiangtan__7__164.tif',
                 show = False,
                 save = False,
                 save_name= 'result'):
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    result = inference_segmentor(model, img)
    png = [cv2.imread(pngfile,-1)]
    if show:
        show_result_pyplot(model, img, result, opacity=0.7) # only to show the result ,no save
    # you can change the opacity of the painted segmentation map in (0, 1].
    if save:
        model.show_result(img, result, out_file= save_name + '.jpg', opacity=0.7)
test_sig_img()
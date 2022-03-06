import time
from collections import OrderedDict

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
import mmcv
from mmseg.core.evaluation import eval_hooks, eval_metrics,intersect_and_union, mean_iou
import cv2
import numpy as np
from PIL import Image
import os

# 可视化还没改
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

def test_sig_img(model,
                 img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__11__50.tif',
                 gt = '',
                 show = False,
                 save = False,
                 save_name= 'result'):
    result = inference_segmentor(model, img)
    # gt有输入，表示对比推理结果和真实标签，输出评价指标
    if gt is not None:
        gt = cv2.imread(gt_file, -1)
        eval_res = eval_metrics(result, gt, 9, ignore_index=10, )  # 为什么一定要输入一个要被忽略的类别呢,感觉没啥影响是因为之前写的0类根本是NAN
        # IoU = intersect_and_union(np.array(result)[0], gt, 9, ignore_index= 10, ) #不知道转成了多维数组就自动多了一个维度，所以取了[0]
        # mIoU  = mean_iou(np.array(result)[0], gt, 9, ignore_index= 10, )
        score = eval_res['aAcc'].tolist()
    if show:
        show_result_pyplot(model, img, result, opacity=0.5) # only to show the result ,no save
    # visualize the results in a new window
    # model.show_result(img, result, show=True)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    if save:
        model.show_result(img, result, out_file='./' + save_name + '.jpg', opacity=0.5)
    return result

# def cal_sig_aAcc(config_file,checkpoint_file,img,gt_file):
    # model = init_segmentor(config_file, checkpoint_file, device='cuda:0') # 这里应该可以注释掉，不需要加载model
    # img = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/images/xiangtan__11__50.tif'  # or img = mmcv.imread(img), which will only load it once
    # gt_file = '/media/xjw/data/seg_cl/myseg_cl/data/new_data/annotations/xiangtan__11__50.tif'
    # gt = cv2.imread(gt_file, -1)
    # result = test_sig_img(model,img)
    # eval_res = eval_metrics(result, gt, 9, ignore_index= 10, ) # 为什么一定要输入一个要被忽略的类别呢,感觉没啥影响是因为之前写的0类根本是NAN
    # gt = gt.tolist()
    # IoU = intersect_and_union(np.array(result)[0], gt, 9, ignore_index= 10, ) #不知道转成了多维数组就自动多了一个维度，所以取了[0]
    # mIoU  = mean_iou(np.array(result)[0], gt, 9, ignore_index= 10, )
    # print("eval_res is : ", eval_res)
    # print("IoU is : ", IoU)
    # print("mIoU is : ", mIoU)
    # score = eval_res['aAcc'].tolist()
    # return score

# 注入口
def cal_data(config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py',
            checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori1_work_dirs/myconfig/iter_80000.pth',
            img_txt_path= '',
            score_txt_path = '',
            gt = False):
    """

    @param gt: False表示不知道新数据的标签数据，只能计算出不确定性；True表示知道标签数据，可以计算出一系列评价指标
    @param txt_path: txt文件中存储了需要被计算不确定性的数据集文件的名称
    """
    filenames = []
    start_time = time.time()
    ROOT = '/media/xjw/data/seg_cl/myseg_cl/data/new_data'
    seq = OrderedDict()
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')
    with open(img_txt_path, 'r') as f1:
        for i in f1.readlines():
            name = i.strip()
            filenames.append(name)
    for idx, ch in enumerate(filenames):
        img = ROOT + '/images/' + ch + '.tif'
        if not gt:
            gt_file = ROOT + '/annotations/' + ch + '.tif'
        else:
            gt_file = ''
        score = test_sig_img(model, img, gt_file)
        print(ch, score, idx, len(filenames))
        seq[ch] = score
        # 查一下aAcc的取值范围
    # with open('aAcc_rank.json', 'w', encoding='utf-8') as f:
    #     json.dump(seq, f, indent=4)
    # f.close()
    with open(score_txt_path, 'a') as f:
        for ch,i in seq.items():
            f.write(ch + ' ' + str(i) + '\n')
    f.close()
    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    # 经过我严密的推理，eval_metrics和mIoU计算的是同一个东西（nonono）
    # 测试单张图片并且可视化
    # @TODO：dataroot怎么处理比较方便呢？
    cal_data(config_file='/home/xjw/Downloads/chongqing/base_data_training/config.py',
             checkpoint_file='/home/xjw/Downloads/chongqing/base_data_training/base_model/iter_10.pth',
             img_txt_path='/home/xjw/Downloads/chongqing/incr_data_learning/icl_model/_2.txt',
             score_txt_path='/home/xjw/Downloads/chongqing/incr_data_learning/icl_model/mIoU.txt',
             gt = True)
    # vis_img_gr()
        
# 11_50,10_205
#{'aAcc': array(95.36328125), 'IoU': array([       nan, 0.83421438,        nan, 0.19274611, 0.        ,
#        0.62170841,        nan, 0.09732143,        nan]), 'Acc': array([         nan, 121.27777778,          nan,  19.92857143,
#                 nan,  97.68807339,          nan,   9.90909091,
#                 nan])}



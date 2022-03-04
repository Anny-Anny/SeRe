import time

from SeRe.Sele.aAcc_based_tools import cal_sig_aAcc
import os
import json
from collections import OrderedDict


def aAcc_data():
    start_time = time.time()
    ROOT = '/media/xjw/data/seg_cl/myseg_cl/data/new_data'
    seq = OrderedDict()
    config_file = '/media/xjw/data/seg_cl/myseg_cl/config/myconfig_before.py'
    checkpoint_file = '/media/xjw/data/seg_cl/mmsegmentation-master/ori1_work_dirs/myconfig/iter_80000.pth'
    filenames = os.listdir(ROOT + '/' + 'images')
    for idx, ch in enumerate(filenames):
        img = ROOT + '/images/' + ch
        gt_file = ROOT + '/annotations/' + ch
        aAcc = cal_sig_aAcc(config_file, checkpoint_file, img, gt_file)
        print(ch, aAcc, idx, len(filenames))
        seq[ch] = aAcc
        # 查一下aAcc的取值范围
    with open('aAcc_rank.json', 'w', encoding='utf-8') as f:
        json.dump(seq, f, indent=4)
    f.close()
    print("total_time: ", time.time() - start_time)


def rank(txt_path,reverse = True):
    """

    @param txt_path:需要进行排列的txt文件
    @param reverse:默认是升序
    """
    data = {}
    new_data = {}

    with open(txt_path, 'r') as f1:
        dataroot = os.path.dirname(txt_path)
        for i in f1.readlines():
            name = i.split(' ')[0].strip()
            score = i.split(' ')[1].strip()
            data[name] = score
    # with open('rank.json', 'r') as f1:
    #     data = json.load(f1)
    with open(dataroot + '/' + 'rank_up.json', 'w', encoding='utf-8') as f:
        data = sorted(data.items(), key=lambda a: (a[1], a[0])) # sorted 返回的是一个元祖类型
        for i in data:
            new_data[i[0]] = i[1]
        json.dump(new_data, f, indent=4)
    f.close()
# rank('/home/xjw/Downloads/chongqing/incr_data_learning/icl_model/xixi.txt')

def gen_txt(json_path, interval = 5):
    """
    生成4等分按照排序的txt文件，作为split参数传入myconfig的traindata
    @param json_path: 输入已经排完序的json文件，用于划分等分的txt
    @TODO:这里是四等分 但是实验的数据是要叠加的
    :return:
    """
    data = {}
    total = []
    dataroot = os.path.dirname(json_path)
    with open(json_path, 'r') as f1:
        data = json.load(f1)
    total = list(data.keys())
    L = len(data)
    n = interval  # 切分成多少份
    step = int(L / n) + 1  # 每份的长度
    idx = 1
    for i in range(0, L, step):
        # print(len(total[i: i + step]))
        filename = '_' + str(idx) + '.txt'
        data_i = total[: i + step]
        with open(dataroot + '/' + filename, 'w') as f:
            for ch in data_i:
                f.write(ch + '\n')
        f.close()
        idx += 1


gen_txt('/home/xjw/Downloads/chongqing/incr_data_learning/icl_model/rank_up.json')



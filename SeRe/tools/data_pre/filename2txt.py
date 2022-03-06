import os

ROOT = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/'


def filename2txt(filepath, txt_path):
    with open(txt_path, 'w+') as f:
        for root, dirs, files in os.walk(filepath):
            for name in files:
                f.write(name.split(".")[0] + '\n')
    f.close()


def txt2filename(txt_path):
    """

    @param txt_path:
    @return: 返回数据集中包含的所有图片的名称
    """
    data = []
    with open(txt_path, 'r') as f:
        for ch in f.readlines():
            data.append(ch.strip())
    return data


if __name__ == '__main__':
    filename2txt('/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan/images/validation',
                 ROOT + 'SeRe/tools/data_pre/val.txt')

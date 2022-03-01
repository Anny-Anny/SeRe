import os

ROOT = '/home/xjw/Downloads/code/mmsegmentation-0.21.0/'
def filename2txt(filepath, txtpath):
    with open(txtpath,'w+') as f:
        for root, dirs, files in os.walk(filepath):
            for name in files:
                f.write(name.split(".")[0] + '\n')
    f.close()

if __name__ == '__main__':
 filename2txt('/home/xjw/Downloads/code/mmsegmentation-0.21.0/data/xiangtan/images/validation',ROOT + 'SeRe/tools/data_pre/val.txt')
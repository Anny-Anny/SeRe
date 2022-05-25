#coding: utf-8
import pynvml
import os
import time
# 功能：检测显存和指定卡号、启动运行
gpu_memory = 40 #预期GPU剩余显存，如果达到这个数字就运行train脚本
trainpythonfile= '/home/dgx/workspace/xjw/mmsegmentation/tools/train.py'
pynvml.nvmlInit()

# gpu_list = [
#     'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
#     'os.environ["CUDA_VISIBLE_DEVICES"] = "1"',
#     'os.environ["CUDA_VISIBLE_DEVICES"] = "2"',
#     'os.environ["CUDA_VISIBLE_DEVICES"] = "3"',
#     'os.environ["CUDA_VISIBLE_DEVICES"] = "4"',
# ]
# gpu_list = [
#     'os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
# ]
# 读写python文件在指定位置插入显卡编号
def writepythonfile(pythonfile,cline,string):
    lines = []
    with open(pythonfile,"r") as y:
        for line in y:
            lines.append(line)
        y.close()
    lines.insert(cline, string+"\n")
    s = ''.join(lines)
    with open(pythonfile,"w") as z:
        z.write(s)
        z.close()
    del lines[:]
# 服务器的多卡gpu测试

while 1:
    num = 0
    gpu = []
    gpu_list = [0,1,2,3,4]
    for i in gpu_list:
        if i == 3:
            print("this is 3")
        if i == 4:
            print("this is 4")
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print('第'+str(i)+'块GPU剩余显存'+str(meminfo.free/(1024**3))+'GB') #第二块显卡剩余显存大小
        if meminfo.free/(1024**2)>=gpu_memory*1024:
            print("!符合剩余" + str(gpu_memory) + "GB显存需求")
            gpu.append(i)
            num += 1
            if num == 2:
                writepythonfile(trainpythonfile, 0, 'import os')
                writepythonfile(trainpythonfile, 1, 'os.environ["CUDA_VISIBLE_DEVICES"] = ' + '\'' + str(gpu[0]) + ',' + str(gpu[1]) + '\'')
                # writepythonfile(trainpythonfile, 1, 'os.environ["CUDA_VISIBLE_DEVICES"] = ' + '\'' + str(gpu[0]) + '\'')
                os.system('bash /home/dgx/workspace/xjw/mmsegmentation/tools/dist_train1.sh')
                break
        else:
            print("不符合剩余"+str(gpu_memory)+"GB显存需求")
        time.sleep(5)
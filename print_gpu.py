#coding: utf-8
import psutil
import pynvml
import os
import time
import yagmail
gpu_memory = 40 #预期GPU剩余显存，如果达到这个数字就运行train脚本
trainpythonfile= '/home/dgx/workspace/xjw/mmsegmentation/tools/train.py'
pynvml.nvmlInit()

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

def get_pid():
    # 获取当前所有进程
    pids = psutil.pids()
    return pids

def in_pids(*aa):
    # print(aa)
    flag = True
    pids = get_pid()
    for i in aa:
        print(i)
        flag = flag and (i in pids)
    return flag
while 1:
    # print(get_pid())
    if (in_pids(53912,53913)):
        print("还在跑，别着急")
    else:
        print("跑完了，吃个鸡腿！")
        # 链接邮箱服务器
        yag = yagmail.SMTP(user="1207327296@qq.com", password="dhslifceafwahjia", host='smtp.qq.com')
        # 邮箱正文
        contents = ['跑完了，吃个鸡腿！',
                    '来自锤锤小甜心', ]
        # 发送邮件
        yag.send('1207327296@qq.com', 'subject', contents)
    time.sleep(60)
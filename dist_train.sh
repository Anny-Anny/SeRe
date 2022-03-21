#!/usr/bin/env bash

CONFIG='/home/dgx/workspace/xjw/mmsegmentation/SeRe/config/dgx/xt/xt_union/myconfig_xt_200.py'
GPUS=2
PORT=${PORT:-29507}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} \
    --resume-from /home/dgx/workspace/xjw/mmsegmentation/tools/work_dirs/xiangtan/union/train_100/epoch_100.pth \
    --work-dir /home/dgx/workspace/xjw/mmsegmentation/tools/work_dirs/xiangtan/union/train_200
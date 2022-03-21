#!/usr/bin/env bash

CONFIG='/home/dgx/workspace/xjw/mmsegmentation/SeRe/config/dgx/myconfig_xt.py'
CHECKPOINT='/home/dgx/workspace/xjw/mmsegmentation/tools/work_dirs/myconfig_xt/.eval_hook/part_1.pkl'
GPUS=4
PORT=${PORT:-29507}
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}

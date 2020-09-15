#!/bin/bash
set -e -x

DATE=`date +%d_%m_%Y`
CODE=./src
DATA=../flute/final
CHECKPOINT=./checkpoints/pretrained_musicnet/lastmodel
DECODER=1
UPDATE=10

EXP=flute-${DECODER}-${UPDATE}
export MASTER_PORT=29500

python ${CODE}/finetune.py \
    --gpu 0,1,2,3 \
    --epochs 60 \
    --data ${DATA} \
    --checkpoint ${CHECKPOINT} \
    --expName ${EXP} \
    --batch-size 16 \
    --lr-decay 0.995 \
    --epoch-len 1000 \
    --num-workers 2 \
    --lr 1e-3 \
    --seq-len 12000 \
    --data-aug \
    --decoder-update ${UPDATE} \
    --decoder ${DECODER}
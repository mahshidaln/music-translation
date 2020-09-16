# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env bash

#Decoders:
#0: Cello
#1: Beethoven Piano 
#2:
#3:
#4:
#5:

DATE=`date +%d_%m_%Y`
CODE=src
DECODER=1
UPDATE=10
OUTPUT=results/flute-${DECODER}-${UPDATE}
DATA=../musicnet/preprocessed/Bach_Solo_Piano

echo "Sampling"
python ${CODE}/data_samples.py --data ${DATA} --output ${OUTPUT}-py  -n 2 --seq 80000

echo "Generating"
python ${CODE}/run_on_files.py --files ${OUTPUT}-py --gpu 0,1,2,3 --batch-size 6 --model checkpoints/pretrained_musicnet/lastmodel_${DECODER}.pth --checkpoint checkpoints/flute-${DECODER}-${UPDATE}/lastmodel --output-next-to-orig --decoders 0 --py

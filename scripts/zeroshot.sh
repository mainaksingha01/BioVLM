#!/bin/bash

# custom config
DATA="PATH/TO/DATA"
TRAINER=ZeroshotCLIP
#TRAINER=DESC
DATASET=$1
CFG=biomed  # rn50, rn101, vit_b32 or vit_b16
SUB=new

python train.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/CoOp/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET}/${SUB} \
--eval-only \
DATASET.SUBSAMPLE_CLASSES ${SUB}
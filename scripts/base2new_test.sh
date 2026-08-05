#!/bin/bash

cd ../..

# custom config
DATA="PATH/TO/DATA"
TRAINER=BioVLM

DATASET=$1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp

LOADEP=50
SUB=new
CFG=biomed  # config file
CTP=end  # class token position (end or middle)
NCTX=4  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False

for SEED in 1 2 3
do
    COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
    MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
    # MODEL_DIR=output/base2new/train_base/tissuemnist/shots_16/CoOp/biomed/seed${SEED}
    DIR=output/base2new/test2_${SUB}/biodata/${COMMON_DIR}
    if [ -d "$DIR" ]; then
        echo "Oops! The results exist at ${DIR} (so skip this job)"
    else
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.SUBSAMPLE_CLASSES ${SUB}
    fi
done
#!/bin/bash

LOSSES=(
    "crossentropy"
    "jaccard"
)
TRAIN_LIST=(
    "train_MD_region_patch_LC1_all_NLCD"
    "train_Chesapeake2013_region_patch_LC1_all_NLCD"
    "train_Chesapeake2014_region_patch_LC1_all_NLCD"
)

MODEL_TYPES=(
    "baseline"
    "extended"
    "extended_bn"
    "extended2_bn"
    "unet1"
)

((TIME_BUDGET=3600*12))
BATCH_SIZE_EXPONENT=4
((BATCH_SIZE=2**$BATCH_SIZE_EXPONENT))
GPU_ID=3
LOSS=${LOSSES[0]}
LEARNING_RATE=0.003
TRAIN_LIST_IDX=0
MODEL_TYPE=${MODEL_TYPES[4]}

TRAIN_PATCH_LIST=/mnt/afs/chesapeake/for-le/Kolya_paper_patch_list/${TRAIN_LIST[${TRAIN_LIST_IDX}]}.txt

EXP_NAME=ForKDD-landcover-batch_size-${BATCH_SIZE}-loss-${LOSS}-lr-${LEARNING_RATE}-train_patch-${TRAIN_LIST_IDX}-model-${MODEL_TYPE}-schedule-stepped
OUTPUT=/mnt/blobfuse/train-output

if [ -d "${OUTPUT}/${EXP_NAME}" ]; then
    echo "Experiment ${OUTPUT}/${EXP_NAME} exists"
    while true; do
        read -p "Do you wish to overwrite this experiment? [y/n]" yn
        case $yn in
            [Yy]* ) rm -rf ${OUTPUT}/${EXP_NAME}; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer y or n.";;
        esac
    done
fi

mkdir -p ${OUTPUT}/${EXP_NAME}/
cp -r *.sh *.py ${OUTPUT}/${EXP_NAME}/

echo ${OUTPUT}/${EXP_NAME}/log.txt

unbuffer python -u train_model_landcover.py \
    --name ${EXP_NAME} \
    --output ${OUTPUT} \
    --gpu ${GPU_ID} \
    --model_type ${MODEL_TYPE} \
    --learning_rate ${LEARNING_RATE} \
    --batch_size ${BATCH_SIZE} \
    --loss ${LOSS} \
    --time_budget ${TIME_BUDGET} \
    --verbose 1 \
    --training_patches ${TRAIN_PATCH_LIST} \
    &> ${OUTPUT}/${EXP_NAME}/log.txt &

# python -u train_model_landcover.py \
#     --name ${EXP_NAME} \
#     --output ${OUTPUT} \
#     --gpu ${GPU_ID} \
#     --learning_rate ${LEARNING_RATE} \
#     --batch_size ${BATCH_SIZE} \
#     --loss ${LOSS} \
#     --time_budget ${TIME_BUDGET} \
#     --verbose 1 \
#     --training_patches ${TRAIN_PATCH_LIST}

#wait;
exit

#done

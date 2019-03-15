#!/bin/bash

ARRAY=(
    "uniform"
    "importance_training"
    "biased_importance_training"
    "approximate_importance_training"
    "rejection"
)
LOSSES=(
    "crossentropy"
    "jaccard"
)
TRAIN_LIST=(
    "train_MD_region_patch_LC1_all_NLCD"
    "train_Chesapeake2013_region_patch_LC1_all_NLCD"
    "train_Chesapeake2014_region_patch_LC1_all_NLCD"
)

TEST_SPLITS=(
    "de_1m_2013"
    "md_1m_2013"
    "md_1m_2015"
    "nc_1m_2014"
    "nj_1m_2013"
    "ny_1m_2013"
    "pa_1m_2013"
    "va_1m_2014"
    "wv_1m_2014"
)

MODEL_TYPES=(
    "baseline"
    "extended"
    "extended_bn"
    "extended2_bn"
    "unet1"
)


BATCH_SIZE_EXPONENT=4
((TIME_BUDGET=3600*5))
((BATCH_SIZE=2**$BATCH_SIZE_EXPONENT))
GPU_ID=0
SAMPLER=${ARRAY[4]}
LOSS=${LOSSES[0]}
LEARNING_RATE=0.003
TRAIN_LIST_IDX=0
MODEL_TYPE=${MODEL_TYPES[4]}


EXP_NAME=ForKDD-landcover-sampler-${SAMPLER}-batch_size-${BATCH_SIZE}-loss-${LOSS}-lr-${LEARNING_RATE}-train_patch-${TRAIN_LIST_IDX}-model-${MODEL_TYPE}-schedule-stepped
OUTPUT=/mnt/blobfuse/train-output
PRED_OUTPUT=/mnt/blobfuse/pred-output/ForKDD_KDD_test_format

if [ ! -f "${OUTPUT}/${EXP_NAME}/final_model.h5" ]; then
    echo "This experiment hasn't been trained! Exiting..."
    exit
fi


if [ -d "${PRED_OUTPUT}/${EXP_NAME}" ]; then
    echo "Experiment output ${PRED_OUTPUT}/${EXP_NAME} exists"
    while true; do
        read -p "Do you wish to overwrite this experiment? [y/n]" yn
        case $yn in
            [Yy]* ) rm -rf ${PRED_OUTPUT}/${EXP_NAME}; break;;
            [Nn]* ) exit;;
            * ) echo "Please answer y or n.";;
        esac
    done
fi

mkdir -p ${PRED_OUTPUT}/${EXP_NAME}/


for TEST_SPLIT in "${TEST_SPLITS[@]}"
do
	echo $TEST_SPLIT
    TEST_CSV=/mnt/afs/code/minibatch/splits/${TEST_SPLIT}_test_split.csv
    echo ${PRED_OUTPUT}/${EXP_NAME}/log_test_${TEST_SPLIT}.txt
    unbuffer python -u test_model_landcover.py \
        --input ${TEST_CSV} \
        --output ${PRED_OUTPUT}/${EXP_NAME}/ \
        --model ${OUTPUT}/${EXP_NAME}/final_model.h5 \
        --gpu ${GPU_ID} \
        &> ${PRED_OUTPUT}/${EXP_NAME}/log_test_${TEST_SPLIT}.txt

    echo ${PRED_OUTPUT}/${EXP_NAME}/log_acc_${TEST_SPLIT}.txt
    unbuffer python -u compute_accuracy.py \
        --input_list ${TEST_CSV} \
        --pred_blob_root ${PRED_OUTPUT}/${EXP_NAME} \
        &> ${PRED_OUTPUT}/${EXP_NAME}/log_acc_${TEST_SPLIT}.txt &
done

wait;

echo "./eval_all_landcover_results.sh ${EXP_NAME}"

exit 0
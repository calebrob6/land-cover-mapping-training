#!/bin/bash

LOSSES=(
    "crossentropy"
    "jaccard"
    "superres"
)


TEST_SPLITS=(
    "md_1m_2013"
    "de_1m_2013"
    "ny_1m_2013"
    "pa_1m_2013"
    "va_1m_2014"
    "wv_1m_2014"
)
TEST_SPLIT="ny_1m_2013"

GPU_ID=0
MODEL_FN="model_115.h5"
MODEL_FN_INST=${MODEL_FN%.*}
USER_MODEL_FILE="output/hr_runs.txt"
#USER_MODEL_FILE="output/hr+sr_runs.txt"

EXP_NAME=ForICCV-landcover-batch_size-16-loss-crossentropy-lr-0.003-model-unet2-schedule-stepped-note-replication_1
TRAIN_OUTPUT=/mnt/blobfuse/train-output/ForICCV
PRED_OUTPUT=/mnt/blobfuse/pred-output/ForICCV

if [ ! -f "${TRAIN_OUTPUT}/${EXP_NAME}/${MODEL_FN}" ]; then
    echo "This experiment hasn't been trained! Exiting..."
    exit
fi


while IFS="," read -r fn idx num remainder
do
    echo ${fn} ${idx} ${num}
    
    EXP_NAME_OUT=${EXP_NAME}-instance-${MODEL_FN_INST}-user_idx-${idx}-model_idx-${num}

    if [ -d "${PRED_OUTPUT}/${EXP_NAME_OUT}" ]; then
        echo "Experiment output ${PRED_OUTPUT}/${EXP_NAME_OUT} exists"
        while true; do
            read -p "Do you wish to overwrite this experiment? [y/n]" yn
            case $yn in
                [Yy]* ) rm -rf ${PRED_OUTPUT}/${EXP_NAME_OUT}; break;;
                [Nn]* ) exit;;
                * ) echo "Please answer y or n.";;
            esac
        done
    fi

    mkdir -p ${PRED_OUTPUT}/${EXP_NAME_OUT}

    echo ${MODEL_FN} > ${PRED_OUTPUT}/${EXP_NAME_OUT}/model_fn.txt
    echo ${fn} ${idx} ${num} ${remainder} >> ${PRED_OUTPUT}/${EXP_NAME_OUT}/model_fn.txt

    echo $TEST_SPLIT
    TEST_CSV=splits/${TEST_SPLIT}_ICCV_test_split.csv
    echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt
    unbuffer python -u test_model_landcover_adapted.py \
        --input ${TEST_CSV} \
        --output ${PRED_OUTPUT}/${EXP_NAME_OUT}/ \
        --model ${TRAIN_OUTPUT}/${EXP_NAME}/${MODEL_FN} \
        --gpu ${GPU_ID} \
        --aug_model ${fn} \
        &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt
        #--superres \

    echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt
    unbuffer python -u compute_accuracy.py \
        --input_list ${TEST_CSV} \
        --pred_blob_root ${PRED_OUTPUT}/${EXP_NAME_OUT} \
        &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt &


done < ${USER_MODEL_FILE}


wait;
exit 0

# for TEST_SPLIT in "${TEST_SPLITS[@]}"
# do
# 	echo $TEST_SPLIT
#     TEST_CSV=splits/${TEST_SPLIT}_ICCV_test_split.csv
#     echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt
#     unbuffer python -u test_model_landcover.py \
#         --input ${TEST_CSV} \
#         --output ${PRED_OUTPUT}/${EXP_NAME_OUT}/ \
#         --model ${TRAIN_OUTPUT}/${EXP_NAME}/${MODEL_FN} \
#         --gpu ${GPU_ID} \
#         &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_test_${TEST_SPLIT}.txt

#     echo ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt
#     unbuffer python -u compute_accuracy.py \
#         --input_list ${TEST_CSV} \
#         --pred_blob_root ${PRED_OUTPUT}/${EXP_NAME_OUT} \
#         &> ${PRED_OUTPUT}/${EXP_NAME_OUT}/log_acc_${TEST_SPLIT}.txt &
# done

# wait;

# echo "./eval_all_landcover_results.sh ${PRED_OUTPUT}/${EXP_NAME_OUT}"
# exit 0
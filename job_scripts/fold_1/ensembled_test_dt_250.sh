#!/bin/bash
#$ -M pmoreira@nd.edu	# Email address for job notification
#$ -m abe				# 
#$ -pe smp 4
#$ -q gpu
#$ -t 21-30
#$ -N pad_ensembled_250
#$ -l gpu=1

# -- variables
export DATASET_GROUP_ID=1
export DATASET_SIZE=250
export BATCH_SIZE=128
export DATASET_PATH=`printf "../datasets/%d" ${DATASET_GROUP_ID}`
export DATASET_NAME=`printf "msh_tanzania_bal-%d-%d" ${DATASET_GROUP_ID} ${DATASET_SIZE}`
export OUTPUT_PATH=`printf "/scratch365/pmoreira/pads_ensembled_test_v3/output_%d" ${DATASET_GROUP_ID}`
#export CUDA_VISIBLE_DEVICES=${SGE_HGR_gpu_card}


# -- additional modules
module load tensorflow

cd ../../src/
python main.py train \
    --sge_task_id ${SGE_TASK_ID} \
    --dataset_path $DATASET_PATH \
    --dataset_name $DATASET_NAME \
    --output_path $OUTPUT_PATH \
    --dataset_size $DATASET_SIZE \
    --device_number $CUDA_VISIBLE_DEVICES \
    --batch_size $BATCH_SIZE


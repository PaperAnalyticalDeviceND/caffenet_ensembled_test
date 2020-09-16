#!/bin/bash
# -- variables


export DATASET_GROUP_ID=3
export DATASET_SIZE=400
export DATASET_PATH=`printf "../datasets/%d/fold_%d" ${DATASET_GROUP_ID} ${DATASET_GROUP_ID} `
export DATASET_NAME=`printf "msh_tanzania_bal-%d" ${DATASET_SIZE}`
#export DATASET_SIZE_LIST=25,50
export DATASET_SIZE_LIST=25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400
export TRAIN_PERCENT=0.7


# -- additional modules
module load tensorflow

cd ../../src/
python downsample_dataset.py \
    --dataset_path $DATASET_PATH \
    --dataset_name $DATASET_NAME \
    --dataset_group_id $DATASET_GROUP_ID \
    --dataset_size_list $DATASET_SIZE_LIST  \
    --train_percent $TRAIN_PERCENT

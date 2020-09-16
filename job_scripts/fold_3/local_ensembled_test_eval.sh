#!/bin/bash
# -- variables
export DATASET_SIZE_LIST=25,50,75,100,125,150,175,200,225,250,275,300,325,350,375,400
export DATASET_GROUP_ID=3
export DRUG_LABEL_FNAME=../datasets/msh_tanzania_blank_drugs.csv
export NUM_SEEDS=30
export PREDICTION_PATH=`printf "/scratch365/pmoreira/pads_ensembled_test_v3/output_%d/predictions" ${DATASET_GROUP_ID}`
export DATASET_NAME=`printf "msh_tanzania_bal-%d" ${DATASET_GROUP_ID}`
export OUTPUT_PATH=`printf "/scratch365/pmoreira/pads_ensembled_test_v3/output_%d/test_results" ${DATASET_GROUP_ID}`

# -- additional modules
module load tensorflow

cd ../../src/
python main.py test_eval \
    --dataset_size_list $DATASET_SIZE_LIST  \
    --dataset_group_id $DATASET_GROUP_ID \
    --drug_label_fname $DRUG_LABEL_FNAME  \
    --num_seeds $NUM_SEEDS \
    --prediction_path $PREDICTION_PATH \
    --dataset_name $DATASET_NAME \
    --output_path $OUTPUT_PATH

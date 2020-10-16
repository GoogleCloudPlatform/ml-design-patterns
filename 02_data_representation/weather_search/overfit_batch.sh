#!/bin/bash

PROJECT=$(gcloud config get-value project)
BUCKET=${PROJECT}-kfpdemo

PACKAGE_PATH="${PWD}/wxsearch"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="wxsearch_overfit_$now"
MODULE_NAME="wxsearch.train_autoencoder"
REGION="us-central1"

# 10 or so images in dataset
gcloud ai-platform jobs submit training $JOB_NAME \
        --package-path $PACKAGE_PATH \
        --module-name $MODULE_NAME \
        --job-dir gs://${BUCKET}/wxsearch/trained \
        --region $REGION \
        --config train.yaml -- --num_layers 4 --pool_size 4 --num_filters 4 --num_dense 50 \
        --input gs://${BUCKET}/wxsearch/data/2019/tfrecord-00000-* \
        --project ${PROJECT} \
        --batch_size 4 --num_steps 1000 --num_checkpoints 4
        
        
        
#        --config hyperparam.yaml -- \
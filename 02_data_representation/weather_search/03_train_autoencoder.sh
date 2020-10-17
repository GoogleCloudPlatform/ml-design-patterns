#!/bin/bash

PROJECT=$(gcloud config get-value project)
BUCKET=${PROJECT}-kfpdemo

PACKAGE_PATH="${PWD}/wxsearch"
now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="wxsearch_train_$now"
MODULE_NAME="wxsearch.train_autoencoder"
REGION="us-central1"

# 10 or so images in dataset
gcloud ai-platform jobs submit training $JOB_NAME \
        --package-path $PACKAGE_PATH \
        --module-name $MODULE_NAME \
        --job-dir gs://${BUCKET}/wxsearch/trained \
        --region $REGION \
        --config train.yaml -- \
        --input gs://${BUCKET}/wxsearch/data/2019/tfrecord-* \
        --project ${PROJECT} \
        --batch_size 4 --num_steps 50000 --num_checkpoints 10
        
        
        
#        --config hyperparam.yaml -- \
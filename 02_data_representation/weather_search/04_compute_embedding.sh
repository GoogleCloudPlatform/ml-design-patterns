#!/bin/bash

PROJECT=$(gcloud config get-value project)
BUCKET="${PROJECT}-kfpdemo"
TABLE="${PROJECT}:advdata.wxembed"
INPUT="gs://${BUCKET}/wxsearch/data/2019/tfrecord-*"
MODEL="gs://${BUCKET}/wxsearch/trained/savedmodel"

echo "Creating embeddings from ${MODEL} on data in ${INPUT} to ${TABLE}"

python3 -m wxsearch.compute_embedding_main \
    --output_table ${TABLE} \
    --savedmodel ${MODEL} \
    --input ${INPUT} \
    --outdir gs://${BUCKET}/wxsearch/tmp \
    --project ${PROJECT}
 
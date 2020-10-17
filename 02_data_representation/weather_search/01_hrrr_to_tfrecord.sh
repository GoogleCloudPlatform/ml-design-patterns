#!/bin/bash

PROJECT=$(gcloud config get-value project)
BUCKET=${PROJECT}-kfpdemo
STARTDATE="20190101"
ENDDATE="20200101"

echo "Creating HRRR records from ${STARTDATE} in ${BUCKET}"

python3 -m wxsearch.hrrr_to_tfrecord \
    --startdate ${STARTDATE} --enddate ${ENDDATE} \
    --outdir gs://${BUCKET}/wxsearch/data/2019 --project ${PROJECT}

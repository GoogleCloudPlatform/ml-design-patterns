# Workflow Pipeline Pattern

**The code in this subdirectory is adapted from the [official TFX taxi pipeline sample](https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline). Refer to that sample for a detailed e2e TFX pipeline. This one is meant to provide a getting started example using fewer components.**

## Overview

This pipeline trains a model on the NOAA hurricane dataset in BigQuery to predict a hurricane's scale number from its wind speed. We've chosen a relatively simple problem here with only one feature to focus on the tooling.

The pipeline code here is designed to run on [AI Platform Pipelines](https://cloud.google.com/ai-platform/pipelines/docs), and the Trainer and Pusher component make use of AI Platform for training and deploying the model.

## Getting started

First, navigate to the `pipeline/configs.py` file and fill in the variables with the names specific to your Google Cloud project.

## Running the pipeline

To run this pipeline, run the commands below from the root of this `workflow_pipeline` directory. First, define the URL of your Docker image:

```
CUSTOM_TFX_IMAGE='gcr.io/' + GOOGLE_CLOUD_PROJECT + '/tfx-pipeline'
```

Next, create your pipeline. Be sure to replace `ENDPOINT` with the URL of your pipelines dashboard in AI Platform, and `PIPELINE_NAME` with the name of your pipeline as specified in `my_pipeline/pipline/configs.py`. Note that you can leave the pipeline name as its current value (`my_pipeline`).

```
tfx pipeline create  \
--pipeline-path=kubeflow_dag_runner.py \
--endpoint={ENDPOINT} \
--build-target-image={CUSTOM_TFX_IMAGE}
```

Finally, create a run of your pipeline:

```
tfx run create --pipeline-name={PIPELINE_NAME} --endpoint={ENDPOINT}
```

You can view your pipeline execution from the dashboard, which is linked from AI Platform Pipelines in your Cloud console.
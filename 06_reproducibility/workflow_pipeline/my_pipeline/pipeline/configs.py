# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE: this is adapted from the official TFX taxi pipeline sample
# You can find it here: https://github.com/tensorflow/tfx/tree/master/tfx/examples/chicago_taxi_pipeline

import os  # pylint: disable=unused-import

# Pipeline name will be used to identify this pipeline
PIPELINE_NAME = 'my_pipeline'

# TODO: replace with your Google Cloud project
GOOGLE_CLOUD_PROJECT='your-cloud-project'

# TODO: replace with the GCS bucket where you'd like to store model artifacts
# Only include the bucket name here, without the 'gs://'
GCS_BUCKET_NAME = 'your-gcs-bucket'

# TODO: set your Google Cloud region below (or use us-central1)
GOOGLE_CLOUD_REGION = 'us-central1'

RUN_FN = 'pipeline.model.run_fn'

TRAIN_NUM_STEPS = 100
EVAL_NUM_STEPS = 100


BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
   '--project=' + GOOGLE_CLOUD_PROJECT,
   '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
   ]

# The rate at which to sample rows from the Chicago Taxi dataset using BigQuery.
# The full taxi dataset is > 120M record.  In the interest of resource
# savings and time, we've set the default for this example to be much smaller.
# Feel free to crank it up and process the full dataset!
_query_sample_rate = 0.0001  # Generate a 0.01% random sample.

# The query that extracts the examples from BigQuery. This sample uses
# a BigQuery public dataset from NOAA
BIG_QUERY_QUERY = """
    SELECT
      usa_wind,
      usa_sshs
    FROM
      `bigquery-public-data.noaa_hurricanes.hurricanes`
    WHERE
      latitude > 19.5
      AND latitude < 64.85
      AND longitude > -161.755
      AND longitude < -68.01
      AND usa_wind IS NOT NULL
      AND longitude IS NOT NULL
      AND latitude IS NOT NULL
      AND usa_sshs IS NOT NULL
      AND usa_sshs > 0
"""


# A dict which contains the training job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.jobs#Job
GCP_AI_PLATFORM_TRAINING_ARGS = {
    'project': GOOGLE_CLOUD_PROJECT,
    'region': 'us-central1',
    # Starting from TFX 0.14, training on AI Platform uses custom containers:
    # https://cloud.google.com/ml-engine/docs/containers-overview
    # You can specify a custom container here. If not specified, TFX will use
    # a public container image matching the installed version of TFX.
    # Set your container name below.
    'masterConfig': {
      'imageUri': 'gcr.io/' + GOOGLE_CLOUD_PROJECT + '/tfx-pipeline'
    },
#     Note that if you do specify a custom container, ensure the entrypoint
#     calls into TFX's run_executor script (tfx/scripts/run_executor.py)
}

# A dict which contains the serving job parameters to be passed to Google
# Cloud AI Platform. For the full set of parameters supported by Google Cloud AI
# Platform, refer to
# https://cloud.google.com/ml-engine/reference/rest/v1/projects.models

GCP_AI_PLATFORM_SERVING_ARGS = {
    'model_name': PIPELINE_NAME,
    'project_id': GOOGLE_CLOUD_PROJECT,
    # The region to use when serving the model. See available regions here:
    # https://cloud.google.com/ml-engine/docs/regions
    'regions': [GOOGLE_CLOUD_REGION],
}

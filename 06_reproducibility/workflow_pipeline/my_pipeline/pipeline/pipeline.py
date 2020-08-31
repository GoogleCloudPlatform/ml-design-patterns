# Lint as: python2, python3
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text, List, Dict, Any
import tensorflow_model_analysis as tfma
import tensorflow as tf

from ml_metadata.proto import metadata_store_pb2
from tfx.components import BigQueryExampleGen  # pylint: disable=unused-import
from tfx.components import CsvExampleGen
from tfx.components import Evaluator
from tfx.components import ExampleValidator
from tfx.components import Pusher
from tfx.components import ResolverNode
from tfx.components import SchemaGen
from tfx.components import StatisticsGen
from tfx.components import Trainer
from tfx.components import Transform
from tfx.components.base import executor_spec
from tfx.components.trainer import executor as trainer_executor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.extensions.google_cloud_ai_platform.pusher import executor as ai_platform_pusher_executor
from tfx.extensions.google_cloud_ai_platform.trainer import executor as ai_platform_trainer_executor
from tfx.orchestration import pipeline
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from tfx.utils.dsl_utils import external_input


def create_pipeline(
    pipeline_name: Text,
    pipeline_root: Text,
    query: Text,
    run_fn: Text,
    train_args: trainer_pb2.TrainArgs,
    eval_args: trainer_pb2.EvalArgs,
    serving_model_dir: Text,
    metadata_connection_config: Optional[
        metadata_store_pb2.ConnectionConfig] = None,
    beam_pipeline_args: Optional[List[Text]] = None,
    ai_platform_training_args: Optional[Dict[Text, Text]] = None,
    ai_platform_serving_args: Optional[Dict[Text, Any]] = None,
) -> pipeline.Pipeline:

  components = []

  # Brings in training data from a BigQuery public dataset
  example_gen = BigQueryExampleGen(query=query)
  components.append(example_gen)

  # Computes statistics over data for visualization and example validation.
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  components.append(statistics_gen)

  # Generates schema based on statistics files.
  schema_gen = SchemaGen(
      statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True)
  components.append(schema_gen)

  # Performs anomaly detection based on statistics and data schema.
  example_validator = ExampleValidator(  # pylint: disable=unused-variable
      statistics=statistics_gen.outputs['statistics'],
      schema=schema_gen.outputs['schema'])
  components.append(example_validator)

  # NOTE: to keep things simple, this pipeline doesn't implement the Transform component
  # Find out more about Transform here: https://www.tensorflow.org/tfx/guide/transform


  # Parameters for the Trainer component
  # This code trains the model on AI Platform
  trainer_args = {
      'run_fn': run_fn,
      'examples': example_gen.outputs['examples'],
      'schema': schema_gen.outputs['schema'],
      'train_args': train_args,
      'eval_args': eval_args,
      'custom_executor_spec':
          executor_spec.ExecutorClassSpec(trainer_executor.GenericExecutor),
  }
  if ai_platform_training_args is not None:
    trainer_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(
                ai_platform_trainer_executor.GenericExecutor
            ),
        'custom_config': {
            ai_platform_trainer_executor.TRAINING_ARGS_KEY:
                ai_platform_training_args,
        }
    })
  trainer = Trainer(**trainer_args)
  components.append(trainer)

  # Get the latest blessed model for model validation.
  model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))
  components.append(model_resolver)

  # Provide metrics for model evaluation
  eval_config = tfma.EvalConfig(
          model_specs=[tfma.ModelSpec(signature_name='serving_default', label_key='usa_sshs')],
          slicing_specs=[tfma.SlicingSpec()],
          metrics_specs = [tfma.MetricsSpec(
              metrics=[tfma.MetricConfig(class_name='MeanSquaredError', threshold=tfma.MetricThreshold(
                  value_threshold=tfma.GenericValueThreshold(upper_bound={'value': 100.0}),
                  change_threshold=tfma.GenericChangeThreshold(
                      direction=tfma.MetricDirection.LOWER_IS_BETTER,
                      absolute={'value': 100.0}
                  )
              
              ))]
          )]
  )
    
  evaluator = Evaluator(
          examples=example_gen.outputs['examples'],
          model=trainer.outputs['model'],
          baseline_model=model_resolver.outputs['model'],
          eval_config=eval_config)
  components.append(evaluator)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed
  # This code is configured to deploy the model to AI Platform
  pusher_args = {
      'model':
          trainer.outputs['model'],
      'model_blessing':
          evaluator.outputs['blessing'],
      'push_destination':
          pusher_pb2.PushDestination(
              filesystem=pusher_pb2.PushDestination.Filesystem(
                  base_directory=serving_model_dir)),
  }
  if ai_platform_serving_args is not None:
    pusher_args.update({
        'custom_executor_spec':
            executor_spec.ExecutorClassSpec(ai_platform_pusher_executor.Executor
                                           ),
        'custom_config': {
            ai_platform_pusher_executor.SERVING_ARGS_KEY:
                ai_platform_serving_args
        },
    })
  pusher = Pusher(**pusher_args)  # pylint: disable=unused-variable
  components.append(pusher)

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components,
      # Setting enable_cache to False is useful for debugging
      # Once you're happy with your code, try setting it to True
      enable_cache=False,
      metadata_connection_config=metadata_connection_config,
      beam_pipeline_args=beam_pipeline_args,
  )

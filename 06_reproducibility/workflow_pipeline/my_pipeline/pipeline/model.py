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


from __future__ import division
from __future__ import print_function

from absl import logging
import tensorflow as tf


def _gzip_reader_fn(filenames):
    """Small utility returning a record reader that can read gzip'ed files."""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern, batch_size=200):
    """Generates a dataset of features and label for training and evaluation.
  """
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features={
            "usa_wind": tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
            "usa_sshs": tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1)
        },
        reader=_gzip_reader_fn,
        label_key='usa_sshs'
    )
    return dataset


TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40

# TFX Trainer will call this function.
def run_fn(fn_args):


    train_dataset = _input_fn(fn_args.train_files, TRAIN_BATCH_SIZE)
    eval_dataset = _input_fn(fn_args.eval_files, EVAL_BATCH_SIZE)

    for i in train_dataset.take(2):
        print(i)

    feature_columns = [tf.feature_column.numeric_column("usa_wind", shape=(1,))]
    input_layers = {
        "usa_wind": tf.keras.layers.Input(name="usa_wind", shape=(1,), dtype=tf.int32)
    }

    # Build the model with the Keras functional API
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    feature_outputs = feature_layer(input_layers)
    dense_1 = tf.keras.layers.Dense(10, activation='relu')(feature_outputs)
    dense_2 = tf.keras.layers.Dense(10, activation='relu')(dense_1)
    output = tf.keras.layers.Dense(1)(dense_2)
    
    model = tf.keras.Model(inputs=[v for v in input_layers.values()], outputs=[output])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])

  
    model.fit(train_dataset, steps_per_epoch=fn_args.train_steps, validation_data=eval_dataset, validation_steps=fn_args.eval_steps)

    model.save(fn_args.serving_model_dir, save_format='tf')
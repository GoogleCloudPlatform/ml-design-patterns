import os
import tensorflow as tf
import tempfile
from datetime import datetime, timedelta
import numpy as np
import argparse
import logging
import shutil
import subprocess
import apache_beam as beam
import random
import sys

from . import train_autoencoder as trainer  

class Predict(beam.DoFn):
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.embedder = None
    
    def process(self, x):
        # Create a model for every worker only once
        # The Model is not pickleable, so it can not be created
        # in the constructor
        if not self.embedder:
            # create embedder
            print('Loading model into worker TensorFlow version = ', tf.__version__)
            model = tf.keras.models.load_model(self.model_dir)
            embed_output = model.get_layer('refc_embedding').output
            self.embedder = tf.keras.Model(model.input, embed_output, name='embedder')
        
        # embed
        result = x.copy()
        refc = tf.expand_dims(tf.expand_dims(x['ref'], 0), -1) # [h,w] to [1, h, w, 1]
        emb = self.embedder.predict(refc)
        result['ref'] = tf.squeeze(emb, axis=0)
        yield result

def convert_types(x):
    result = {}
    print(x)
    for key in ['size', 'ref']:
        result[key] = x[key].numpy().tolist()
    for key in ['time', 'valid_time']:
        # 'b'2019-09-18T11:00:00'' to 2019-09-18 11:00:00
        result[key] = str(x[key].numpy()).replace('T', ' ').replace('b','').replace("'", '')
    return result

def run_job(options):
    # create objects we need
    schema = {'fields': [
        {'name': 'size', 'type': 'INTEGER', 'mode': 'REPEATED'},
        {'name': 'ref', 'type': 'FLOAT', 'mode': 'REPEATED'},
        {'name': 'time', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
        {'name': 'valid_time', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
    ]}
       
    # start the pipeline
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    with beam.Pipeline(options['runner'], options=opts) as p:
        # read examples
        (
          p
          | 'read_tfr' >> beam.io.tfrecordio.ReadFromTFRecord(os.path.join(options['input']))
          | 'parse_tfr' >> beam.Map(trainer.parse_tfrecord)
          | 'compute_embed' >> beam.ParDo(Predict(options['savedmodel']))
          | 'convert_types' >> beam.Map(convert_types)
          | 'write_bq' >> beam.io.gcp.bigquery.WriteToBigQuery(
              table=options['output_table'], schema=schema,
              write_disposition=beam.io.gcp.bigquery.BigQueryDisposition.WRITE_TRUNCATE)
        )

def main(args):
    parser = argparse.ArgumentParser(
      description='Create embeddings of TF records')
    parser.add_argument(
      '--project',
      default='',
      help='Specify GCP project to bill to run on cloud')
    parser.add_argument(
      '--output_table', required=True, help='PROJECT:dataset.table')
    parser.add_argument(
      '--savedmodel', required=True, help='location of saved autoencoder model')
    parser.add_argument(
      '--input', required=True, help='TF Record pattern')
    parser.add_argument(
      '--outdir', required=True, help='For staging etc.')

    
    # parse command-line args and add a few more
    logging.basicConfig(level=getattr(logging, 'INFO', None))
    options = parser.parse_args().__dict__
    outdir = options['outdir']
    options.update({
      'staging_location':
          os.path.join(outdir, 'tmp', 'staging'),
      'temp_location':
          os.path.join(outdir, 'tmp'),
      'job_name':
          'wxsearch-' + datetime.now().strftime('%y%m%d-%H%M%S'),
      'teardown_policy':
          'TEARDOWN_ALWAYS',
      'max_num_workers':
          5,
      'machine_type':
          'n1-standard-2',
      'region':
          'us-central1',
      'setup_file':
          os.path.join(os.path.dirname(os.path.abspath(__file__)), '../setup.py'),
      'save_main_session':
          False,
    })

    if not options['project']:
        print('Launching local job ... hang on')
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir)
        options['runner'] = 'DirectRunner'
    else:
        print('Launching Dataflow job {} ... hang on'.format(options['job_name']))
        try:
            subprocess.check_call('gsutil -m rm -r {}'.format(outdir).split())
        except:  # pylint: disable=bare-except
            pass
        options['runner'] = 'DataflowRunner'

    print('Local TensorFlow version = ', tf.__version__)
    run_job(options)

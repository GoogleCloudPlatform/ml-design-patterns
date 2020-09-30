
import os
import xarray as xr
import tensorflow as tf
import tempfile
import cfgrib
from datetime import datetime, timedelta
import numpy as np
import argparse
import logging
import shutil
import subprocess
import apache_beam as beam
import random

def _array_feature(value, min_value, max_value):
    """Wrapper for inserting ndarray float features into Example proto."""
    value = np.nan_to_num(value.flatten()) # nan, -inf, +inf to numbers
    value = np.clip(value, min_value, max_value) # clip to valid
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tfrecord(filename):
    with tempfile.TemporaryDirectory() as tmpdirname:
        TMPFILE="{}/read_grib".format(tmpdirname)
        tf.io.gfile.copy(filename, TMPFILE, overwrite=True)
        ds = xr.open_dataset(TMPFILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': 'unknown', 'stepType': 'instant'}})
   
        # create a TF Record with the raw data
        tfexample = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'ref': _array_feature(ds.data_vars['refc'].data, min_value=0, max_value=60),
        }))
        return tfexample.SerializeToString()

def generate_filenames(startdate: str, enddate: str):
    start_dt = datetime.strptime(startdate, '%Y%m%d')
    end_dt = datetime.strptime(enddate, '%Y%m%d')
    dt = start_dt
    while dt < end_dt:
        # gs://high-resolution-rapid-refresh/hrrr.20200811/conus/hrrr.t04z.wrfsfcf00.grib2
        f = '{}/hrrr.{:4}{:02}{:02}/conus/hrrr.t{:02}z.wrfsfcf00.grib2'.format(
                'gs://high-resolution-rapid-refresh',
                dt.year, dt.month, dt.day, dt.hour)
        dt = dt + timedelta(hours=1)
        yield f

def run_job(options):
    # start the pipeline
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    with beam.Pipeline(options['runner'], options=opts) as p:
        # create examples
        examples = (
          p
          | 'hrrr_files' >> beam.Create(
              generate_filenames(options['startdate'], options['enddate']))
          | 'create_tfr' >>
          beam.FlatMap(lambda x: create_tfrecord(x))
        )

        # shuffle the examples so that each small batch doesn't contain
        # highly correlated records
        #examples = (examples
        #      | 'reshuffleA' >> beam.Map(
        #          lambda t: (random.randint(1, 1000), t))
        #      | 'reshuffleB' >> beam.GroupByKey()
        #      | 'reshuffleC' >> beam.FlatMap(lambda t: t[1]))

        # write out tfrecords
        _ = (examples
              | 'write_tfr' >> beam.io.tfrecordio.WriteToTFRecord(
                  os.path.join(options['outdir'], 'tfrecord')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='Create training/eval files for lightning prediction')
    parser.add_argument(
      '--project',
      default='',
      help='Specify GCP project to bill to run on cloud')
    parser.add_argument(
      '--outdir', required=True, help='output dir. could be local or on GCS')
  
    parser.add_argument(
      '--startdate',
      type=str,
      required=True,
      help='eg 20200915')
    parser.add_argument(
      '--enddate',
      type=str,
      required=True,
      help='eg 20200916 -- this is exclusive')
    
    
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
          20,
      'machine_type':
          'n1-standard-8',
      'region':
          'us-central1',
      'setup_file':
          os.path.join(os.path.dirname(os.path.abspath(__file__)), '../setup.py'),
      'save_main_session':
          True,
      # 'sdk_location':
      #    './local/beam/sdks/python/dist/apache-beam-2.12.0.tar.gz'
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

    run_job(options)


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
import sys

LEVEL_TYPE = 'atmosphere'  # unknown?

def _array_feature(value, min_value, max_value):
    if isinstance(value, type(tf.constant(0))): # if value is tensor
        value = value.numpy() # get value of tensor
 
    """Wrapper for inserting ndarray float features into Example proto."""
    value = np.nan_to_num(value.flatten()) # nan, -inf, +inf to numbers
    value = np.clip(value, min_value, max_value) # clip to valid
    logging.info('Range of image values {} to {}'.format(np.min(value), np.max(value)))
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _string_feature(value):
    return _bytes_feature(value.encode('utf-8'))

def create_tfrecord(filename):
    try:
        with tempfile.TemporaryDirectory() as tmpdirname:
            TMPFILE="{}/read_grib".format(tmpdirname)
            tf.io.gfile.copy(filename, TMPFILE, overwrite=True)
            ds = xr.open_dataset(TMPFILE, engine='cfgrib', backend_kwargs={'filter_by_keys': {'typeOfLevel': LEVEL_TYPE, 'stepType': 'instant'}})

            # create a TF Record with the raw data
            refc = ds.data_vars['refc']
            size = np.array([ds.data_vars['refc'].sizes['y'], ds.data_vars['refc'].sizes['x']])
            tfexample = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'size': tf.train.Feature(int64_list=tf.train.Int64List(value=size)),
                        'ref': _array_feature(refc.data, min_value=0, max_value=60),
                        'time': _string_feature(str(refc.time.data)[:19]),
                        'valid_time': _string_feature(str(refc.valid_time.data)[:19])
            }))
            yield tfexample.SerializeToString()
    except:
        e = sys.exc_info()[0]
        logging.error(e)

def create_dummy_tfrecord(filename):
    print(filename)
    with tempfile.TemporaryDirectory() as tmpdirname:
        TMPFILE="{}/read_grib".format(tmpdirname)
        tf.io.gfile.copy(filename, TMPFILE, overwrite=True)

        tfexample = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'size': tf.train.Feature(float_list=tf.train.FloatList(value=[3.0, 4.0])),
        }))
        yield tfexample.SerializeToString()

    
def generate_filenames(startdate: str, enddate: str):
    start_dt = datetime.strptime(startdate, '%Y%m%d')
    end_dt = datetime.strptime(enddate, '%Y%m%d')
    logging.info('Hourly records from {} to {}'.format(start_dt, end_dt))
    dt = start_dt
    while dt < end_dt:
        # gs://high-resolution-rapid-refresh/hrrr.20200811/conus/hrrr.t04z.wrfsfcf00.grib2
        f = '{}/hrrr.{:4}{:02}{:02}/conus/hrrr.t{:02}z.wrfsfcf00.grib2'.format(
                'gs://high-resolution-rapid-refresh',
                dt.year, dt.month, dt.day, dt.hour)
        dt = dt + timedelta(hours=1)
        yield f
                 
def generate_shuffled_filenames(startdate: str, enddate: str):
    """
    shuffle the files so that a batch of records doesn't contain highly correlated entries
    """
    filenames = [f for f in generate_filenames(startdate, enddate)]
    np.random.shuffle(filenames)
    return filenames

def run_job(options):
    # start the pipeline
    opts = beam.pipeline.PipelineOptions(flags=[], **options)
    with beam.Pipeline(options['runner'], options=opts) as p:
        # create examples
        examples = (
          p
          | 'hrrr_files' >> beam.Create(
              generate_shuffled_filenames(options['startdate'], options['enddate']))
          | 'create_tfr' >>
          beam.FlatMap(lambda x: create_tfrecord(x))
        )

        # write out tfrecords
        _ = (examples
              | 'write_tfr' >> beam.io.tfrecordio.WriteToTFRecord(
                  os.path.join(options['outdir'], 'tfrecord')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='Create TF Records from HRRR refc files')
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
          'n1-standard-2',
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

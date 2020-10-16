import tensorflow as tf
import numpy as np
import logging
import hypertune
import argparse
import shutil
import os


def parse_tfrecord(example_data):
    parsed = tf.io.parse_single_example(example_data, {
        'size': tf.io.VarLenFeature(tf.int64),
        'ref': tf.io.VarLenFeature(tf.float32),
        'time': tf.io.FixedLenFeature([], tf.string),
        'valid_time': tf.io.FixedLenFeature([], tf.string)
     })
    parsed['size'] = tf.sparse.to_dense(parsed['size'])
    parsed['ref'] = tf.reshape(tf.sparse.to_dense(parsed['ref']), (1059, 1799))/60. # 0 to 1
    return parsed

def read_dataset(pattern):
    filenames = tf.io.gfile.glob(pattern)
    ds = tf.data.TFRecordDataset(filenames, compression_type=None, buffer_size=None, num_parallel_reads=None)
    return ds.prefetch(tf.data.experimental.AUTOTUNE).map(parse_tfrecord)


def create_model(nlayers=4, poolsize=4, numfilters=5, num_dense=0):
    input_img = tf.keras.Input(shape=(1059, 1799, 1), name='refc_input')

    x = tf.keras.layers.Cropping2D(cropping=((17, 18),(4, 3)), name='cropped')(input_img)
    last_pool_layer = None
    for layerno in range(nlayers):
        x = tf.keras.layers.Conv2D(2**(layerno + numfilters), poolsize, activation='relu', padding='same', name='encoder_conv_{}'.format(layerno))(x)
        last_pool_layer = tf.keras.layers.MaxPooling2D(poolsize, padding='same', name='encoder_pool_{}'.format(layerno))
        x = last_pool_layer(x)
    output_shape = last_pool_layer.output_shape[1:]
    
    if num_dense == 0:
        # flatten to create the embedding
        x = tf.keras.layers.Flatten(name='refc_embedding')(x)
        embed_size = output_shape[0] * output_shape[1] * output_shape[2]
        if embed_size > 1024:
            print("Embedding size={} is too large".format(embed_size))
            return None, embed_size
    else:
        # flatten, send through dense layer to create the embedding
        x = tf.keras.layers.Flatten(name='encoder_flatten')(x)
        x = tf.keras.layers.Dense(num_dense, name='refc_embedding')(x)
        x = tf.keras.layers.Dense(output_shape[0] * output_shape[1] * output_shape[2], name='decoder_dense')(x)
        embed_size = num_dense
        
    x = tf.keras.layers.Reshape(output_shape, name='decoder_reshape')(x)
    for layerno in range(nlayers):
        x = tf.keras.layers.Conv2D(2**(nlayers-layerno-1 + numfilters), poolsize, activation='relu', padding='same', name='decoder_conv_{}'.format(layerno))(x)
        x = tf.keras.layers.UpSampling2D(poolsize, name='decoder_upsamp_{}'.format(layerno))(x)
    before_padding_layer = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', name='before_padding')
    x = before_padding_layer(x)
    htdiff = 1059 - before_padding_layer.output_shape[1]
    wddiff = 1799 - before_padding_layer.output_shape[2]
    if htdiff < 0 or wddiff < 0:
        print("Invalid architecture: htdiff={} wddiff={}".format(htdiff, wddiff))
        return None, 9999
    decoded = tf.keras.layers.ZeroPadding2D(padding=((htdiff//2,htdiff - htdiff//2),
                                                     (wddiff//2,wddiff - wddiff//2)), name='refc_reconstructed')(x)

    autoencoder = tf.keras.Model(input_img, decoded, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.LogCosh()) #loss='mse')
    if autoencoder.count_params() > 1000*1000: # 1 million      
        print("Autoencoder too large: {} params".format(autoencoder.count_params()))
        return None, autoencoder.count_params()
    
    return autoencoder, embed_size

class HptCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.hpt = hypertune.HyperTune()
    
    def on_epoch_end(self, epoch, logs):
        self.hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='final_loss',
            metric_value=logs['loss'],   #history.history['loss'][-1],
            global_step=epoch
        )

def run_job(opts):
    def input_and_label(rec):
        return rec['ref'], rec['ref']
    ds = read_dataset(opts['input']).map(input_and_label).batch(opts['batch_size']).repeat()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(opts['job_dir'], 'checkpoints'))
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        autoencoder, error = create_model(opts['num_layers'], opts['pool_size'], opts['num_filters'], opts['num_dense'])
       
        if autoencoder:    
            print(autoencoder.summary())
            history = autoencoder.fit(ds, steps_per_epoch=opts['num_steps']//opts['num_checkpoints'],
                              epochs=opts['num_checkpoints'], shuffle=True, callbacks=[checkpoint, HptCallback()])
    
            autoencoder.save(os.path.join(opts['job_dir'], 'savedmodel'))
        else:
            HptCallback().on_epoch_end(1, {'loss': error})
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
      description='Train an autoencoder')
    parser.add_argument(
      '--project',
      default='',
      help='Specify GCP project to bill to run on cloud')
    parser.add_argument(
      '--job-dir', required=True, help='output dir. could be local or on GCS')
    parser.add_argument(
      '--input', required=True, help='input pattern. eg: gs://ai-analytics-solutions-kfpdemo/wxsearch/data/tfrecord-*')
    parser.add_argument(
      '--batch_size', default=2, help='batch size for training', type=int)
    parser.add_argument(
      '--num_steps', default=12, help='total number of steps for training', type=int)
    parser.add_argument(
      '--num_checkpoints', default=3, help='number of steps for training', type=int)
    parser.add_argument(
      '--num_layers', default=4, help='number of conv layers in model', type=int)    
    parser.add_argument(
      '--pool_size', default=4, help='size of upscaling/downscaling kernel', type=int)    
    parser.add_argument(
      '--num_filters', default=4, help='efficiency of representation of a tile', type=int)    
    parser.add_argument(
      '--num_dense', default=50, help='size of embedding if you want a dense layer. Specify 0 to use conv layers only', type=int)    
    
     
    # parse command-line args and add a few more
    logging.basicConfig(level=getattr(logging, 'INFO', None))
    tf.debugging.set_log_device_placement(True)
    options = parser.parse_args().__dict__

    outdir = options['job_dir']
    if not options['project']:
        print('Removing local output directory {} ... hang on'.format(outdir))
        shutil.rmtree(outdir, ignore_errors=True)
        os.makedirs(outdir)
    else:
        print('Removing GCS output directory {} ... hang on'.format(outdir))
        try:
            subprocess.check_call('gsutil -m rm -r {}'.format(outdir).split())
        except:  # pylint: disable=bare-except
            pass

    run_job(options)

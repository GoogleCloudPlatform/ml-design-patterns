import tensorflow as tf
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

def create_model(nlayers=4, poolsize=4, embed_size=256):
    input_img = tf.keras.Input(shape=(1059, 1799, 1), name='refc_input')

    x = tf.keras.layers.Cropping2D(cropping=((17, 18),(4, 3)), name='cropped')(input_img)
    last_pool_layer = None
    for layerno in range(nlayers):
        x = tf.keras.layers.Conv2D(2**(nlayers-layerno + 3), 5, activation='relu', padding='same', name='encoder_conv_{}'.format(layerno))(x)
        last_pool_layer = tf.keras.layers.MaxPooling2D(poolsize, padding='same', name='encoder_pool_{}'.format(layerno))
        x = last_pool_layer(x)
    output_shape = last_pool_layer.output_shape[1:]
    
    # flatten input
    x = tf.keras.layers.Flatten(name='encoder_flatten')(x)
    x = tf.keras.layers.Dense(embed_size, name='refc_embedding')(x)
    
    # decode, but starting with the shape of the last preflattened layer
    x = tf.keras.layers.Dense(output_shape[0] * output_shape[1] * output_shape[2], name='decoder_input')(x)
    x = tf.keras.layers.Reshape(output_shape, name='decoder_reshape')(x)
    for layerno in range(nlayers):
        x = tf.keras.layers.Conv2D(2**(layerno + 4), 5, activation='relu', padding='same', name='decoder_conv_{}'.format(layerno))(x)
        x = tf.keras.layers.UpSampling2D(poolsize, name='decoder_upsamp_{}'.format(layerno))(x)
    x = tf.keras.layers.Conv2D(1, 3, activation='relu', padding='same', name='before_padding')(x)
    decoded = tf.keras.layers.ZeroPadding2D(padding=((17,18),(4,3)), name='refc_reconstructed')(x)

    autoencoder = tf.keras.Model(input_img, decoded, name='autoencoder')
    autoencoder.compile(optimizer='adam', loss=tf.keras.losses.LogCosh()) #loss='mse')
    return autoencoder

def run_job(opts):
    def input_and_label(rec):
        return rec['ref'], rec['ref']
    ds = read_dataset(opts['input']).map(input_and_label).batch(opts['batch_size']).repeat()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(os.path.join(opts['job_dir'], 'checkpoints'))
    
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        autoencoder = create_model(opts['num_layers'], opts['pool_size'])
        print(autoencoder)
        history = autoencoder.fit(ds, steps_per_epoch=opts['num_steps']//opts['num_checkpoints'],
                              epochs=opts['num_checkpoints'], shuffle=True, callbacks=[checkpoint])
    
        autoencoder.save(os.path.join(opts['job_dir'], 'savedmodel'))
        
        # report final metric to hyperparameter tuner
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='final_loss',
            metric_value=history.history['loss'][-1],
            global_step=1
        )
    
    
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
      '--embed_size', default=256, help='size of embedding', type=int)    
    
     
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

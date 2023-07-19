'''
Copyright 2023 innodisk Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
 
Author: 
    Hueiru, hueiru_chen@innodisk.com, innodick Inc
'''

'''
Quantize the floating-point model
'''

'''
Author: Mark Harvey
'''


import argparse
import os
from pickletools import optimize
import shutil
import sys

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.optimizers import Adam

DIVIDER = '-----------------------------------------'

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def parser(data_record):
    ''' TFRecord parser '''

    feature_dict = {
      'label' : tf.io.RaggedFeature(dtype=tf.int64),
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width' : tf.io.FixedLenFeature([], tf.int64),
      'chans' : tf.io.FixedLenFeature([], tf.int64),
      'image' : tf.io.FixedLenFeature([], tf.string)
    }
    sample = tf.io.parse_single_example(data_record, feature_dict)
    label = tf.cast(sample['label'], tf.int32)

    h = tf.cast(sample['height'], tf.int32)
    w = tf.cast(sample['width'], tf.int32)
    c = tf.cast(sample['chans'], tf.int32)
    image = tf.io.decode_image(sample['image'], channels=3)
 
    image = tf.reshape(image,[h,w,3])

    return image, label

def normalize(x,y):
    '''
    Image normalization
    Args:     Image and label
    Returns:  normalized image and unchanged label
    '''
    # Convert to floating-point & scale to range 0.0 -> 1.0
    x = tf.cast(x, tf.float32) * (1. / 255)
    return x, y


def resize_crop(x, y, h, w):
    '''
    Image resize & random crop
    Args:     Image and label
    Returns:  augmented image and unchanged label
    '''
    rh = int(h *1.2)
    rw = int(w *1.2)
    x = tf.image.resize(x, (rh,rw), method='bicubic')
    x = tf.image.random_crop(x, [h, w, 3], seed=42)
    return x,y


def input_fn_quant(tfrec_dir,batchsize,height,width):
    '''
    Dataset creation and augmentation for quantization
    The TFRecord file(s) must have > 1000 images
    '''
    tfrecord_files = tf.data.Dataset.list_files('{}*.tfrecord'.format(tfrec_dir), shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: resize_crop(x,y,h=height,w=width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def input_fn_test(tfrec_dir, batchsize, height, width):
    '''
    Dataset creation and augmentation for test
    '''
    print(tfrec_dir)
    tfrecord_files = tf.data.Dataset.list_files('{}*.tfrecord'.format(tfrec_dir), shuffle=False)
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x,y: resize_crop(x, y, h=height, w=width), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batchsize, drop_remainder=False)
    dataset = dataset.map(normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def parse_model_path(model_path):
    model_name = ""
    model_json = ""
    for m in range(len(model_path)):
        if '.h5' in model_path[m]:
            model_name = model_path[m]
        elif 'json' in model_path[m]:
            model_json = model_path[m]
    return model_name, model_json

def quant_model(model_path, quant_model, batchsize, tfrec_dir, evaluate):
    '''
    Quantize the floating-point model
    Save to HDF5 file
    '''

    model_name , model_json = parse_model_path(model_path)

    ## load the floating point trained model
    if model_json:

        with open(model_json, 'r') as json_file:
            model_json = json_file.read()   
        model = model_from_json(model_json)
        model.load_weights(model_name)
    else: 
        model = load_model(model_name)

    print("model input shape: {}".format(model.input_shape))
    
    # get input dimensions of the floating-point model
    height = model.input_shape[1]
    width = model.input_shape[2]
  
    # make TFRecord dataset and image processing pipeline
    quant_dataset = input_fn_quant(tfrec_dir, batchsize, height, width)

    # run quantization
    quantizer = vitis_quantize.VitisQuantizer(model)
    quantized_model = quantizer.quantize_model(calib_dataset=quant_dataset)

    # saved quantized model
    quantized_model.save(quant_model)
    print('Saved quantized model to',quant_model)

    if (evaluate):
        '''
        Evaluate quantized model
        '''
        print('\n'+DIVIDER)
        print ('Evaluating quantized model..')
        print(DIVIDER+'\n')

        test_dataset = input_fn_test(tfrec_dir, batchsize, height, width)

        # quantized_model.compile(optimizer=Adam(),
        #                         loss='sparse_categorical_crossentropy',
        #                         metrics=['accuracy'])

        quantized_model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        
   
        scores = quantized_model.evaluate(test_dataset,
                                          verbose=0)

        print('Quantized model accuracy: {0:.4f}'.format(scores[1]*100),'%')
        print('\n'+DIVIDER)

    return

def main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model',  type=str, nargs='+',          help='Full path of floating-point model. Or model weight and model config.json')
    ap.add_argument('-q', '--quant_model',  type=str, default='./quantize_model.h5', help='Full path of quantized model. Default is build/quant_model/q_model.h5')
    ap.add_argument('-b', '--batchsize',    type=int, default=1,    help='Batchsize for quantization. Default is 50')
    ap.add_argument('-tfdir', '--tfrec_dir',type=str, default='./', help='Full path to folder containing TFRecord files. Default is build/tfrecords')
    ap.add_argument('-e', '--evaluate',     action='store_true',    help='Evaluate floating-point model if set. Default is no evaluation.')
    args = ap.parse_args()  

    print('\n------------------------------------')
    print('TensorFlow version : ',tf.__version__)
    print(sys.version)
    print('------------------------------------')
    print ('Command line options:')
    print (' --model        : ', args.model)
    print (' --quant_model  : ', args.quant_model)
    print (' --batchsize    : ', args.batchsize)
    print (' --tfrec_dir    : ', args.tfrec_dir)
    print (' --evaluate     : ', args.evaluate)
    print('------------------------------------\n')
    
    quant_model(args.model, args.quant_model, args.batchsize, args.tfrec_dir, args.evaluate)


if __name__ ==  "__main__":
    main()
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
    Hueiru hueiru_chen@innodisk.com, innodick Inc
'''

'''
Each TFRecord contains 5 fields:

- label
- height
- width
- channels
- image - PNG encoded

'''

from cProfile import label
import os
import argparse
from tqdm import tqdm
import numpy as np

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
import glob


DIVIDER = '-----------------------------------------'

def _bytes_feature(value):
  '''Returns a bytes_list from a string / byte'''
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    '''Returns a float_list from a float / double'''
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    ''' Returns an int64_list from a bool / enum / int / uint '''
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _calc_num_shards(img_list, img_shard):
    ''' calculate number of shards'''
    last_shard =  len(img_list) % img_shard
    if last_shard != 0:
        num_shards =  (len(img_list) // img_shard) + 1
    else:
        num_shards =  (len(img_list) // img_shard)
    return last_shard, num_shards

def write_tfrec(tfrec_filename, img_list, label_list):
    ''' write TFRecord file '''
    with tf.io.TFRecordWriter(tfrec_filename) as writer:
        i = 0
        for img, _label in zip(img_list ,label_list):
            print("tf_data: {}".format(i), end='\r')
            i +=1
            filePath = os.path.join(img)
            image = tf.io.read_file(filePath)
            image_shape = tf.image.decode_png(image).shape
            _label = _label
            # features dictionary
            feature_dict = {
                'label' : _int64_feature(_label),
                'height': _int64_feature(image_shape[0]),
                'width' : _int64_feature(image_shape[1]),
                'chans' : _int64_feature(image_shape[2]),
                'image' : _bytes_feature(image)
            }

            # Create Features object
            features = tf.train.Features(feature = feature_dict)

            # create Example object
            tf_example = tf.train.Example(features=features)

            # serialize Example object into TFRecord file
            writer.write(tf_example.SerializeToString())

    return

def make_tfrec(dataset, tfrec_dir, img_shard, tf_name, framework):
  
    image_list = []
    label_list = []
    if framework == "obj":
        with open(dataset, 'r') as f:
            _f = [_f.rstrip('\n') for _f in f]

            for line in _f:
                name_end = line.find(' ')
                
                bbox_label = line.find(' ')
                
                if bbox_label == -1:
                    ### not regular solution. Because file name would lose 'g' when return -1 ###
                    file_name = line[:name_end]
                    file_name = file_name + 'g'
                    _bbox = []
                else:
                    file_name = line[:name_end]
                    _bbox_label = line[bbox_label+1:]
                    _bbox_label_split = _bbox_label.split(' ')
                    
                    _bbox = []
                    for l in range(len(_bbox_label_split)):
                        for _b in (_bbox_label_split[l].split(',')):
                            _bbox.append(int(_b))
                
                image_list.append(file_name)
                label_list.append(_bbox)
    
    elif framework == "class":
        f = open('./classes.txt')
        labels = f.readlines()
        _dict = glob.glob(dataset+'/*')
        for label in _dict:
            images = glob.glob(label+'/*')
            for i in images:
                image_list.append(i)
                for l in labels:
                    if l.rstrip('\n') in label:
                        label_list.append(labels.index(l))
    else:
        print("Please set a framework type")
    
    last_shard, num_shards = _calc_num_shards(image_list, img_shard)
    print (num_shards,'TFRecord files will be created.')
    
    start = 0
    for i in tqdm(range(num_shards)):
        tfrec_filename = tf_name+'_'+str(i)+'.tfrecord'
        write_path = os.path.join(tfrec_dir, tfrec_filename)
        if (i == num_shards-1):
            write_tfrec(write_path, image_list[start:], label_list[start:])
        else:
            end = start + img_shard
            write_tfrec(write_path, image_list[start:end], label_list[start:end])
            start = end

    return

def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--dataset',     type=str, help='Path of dataset list(yolo) or path(class)')
    ap.add_argument('-t', '--tfrec_dir',   type=str, default='./', help='Path to TFRecord files')
    ap.add_argument('-s', '--img_shard',   type=int, default=2000,  help='Number of images per shard. Default is 1000') 
    ap.add_argument('-n', '--tfrec_name',  type=str, default='train', help='tfrecord name') 
    ap.add_argument('-f', '--framework',   type=str, help='framework:class, obj')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('DATASET PREPARATION STARTED..')
    print('Command line options:')
    print (' --dataset      : ', args.dataset)
    print (' --tfrec_dir    : ', args.tfrec_dir)
    print (' --img_shard    : ', args.img_shard)
    print (' --tfrec_name   : ', args.tfrec_name)
    print (' --framework    : ', args.framework)

    make_tfrec(args.dataset, args.tfrec_dir, args.img_shard, args.tfrec_name, args.framework)

if __name__ == '__main__':
    run_main()
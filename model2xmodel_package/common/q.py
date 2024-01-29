# Copyright 2023 innodisk Inc.
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
# Author: 
#   Hueiru, hueiru_chen@innodisk.com, innodisk Inc
#   Wilson, wilson_yeh@innodisk.com, innodick Inc

import cv2
import numpy as np

calib_image_list = 'train.txt'
calib_batch_size = 1

input_node_name = open('model_input_node_name.txt').readline().splitlines()
image_w, image_h = open('model_input_size.txt').readline().split()

#normalization factor to scale image 0-255 values to 0-1 #DB
NORM_FACTOR = 255.0 # could be also 256.0

def ScaleTo1(x_test):
    x_test  = np.asarray(x_test)
    x_test = x_test.astype(np.float32)
    new_x_test = x_test / NORM_FACTOR
    return new_x_test

def calib_input(iter):
  images = []
  line = open(calib_image_list).readlines()
  for index in range(0, calib_batch_size):
    curline = line[iter * calib_batch_size + index]
    calib_image_name = curline.strip()
    
    image = cv2.imread(calib_image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    
    custom_image = cv2.resize(image, (int(image_w),int(image_h)), interpolation=cv2.INTER_NEAREST)
    image2 = np.array(custom_image)
    print(image2.shape)
    image2 = ScaleTo1(image2)
    images.append(image2)
    print(image2.shape)
    print("Iteration number : {} and index number {} and  file name  {} ".format(iter, index, line))
  return {input_node_name[0]: images} #需換成自己的input node name

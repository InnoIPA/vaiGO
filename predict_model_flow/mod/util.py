# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import cv2
import numpy as np
import logging
import json

# try:
import tensorflow as tf
# except:
# 	pass

def preprocess_fn(image, rs):
	'''
	Image pre-processing.
	Rearranges from BGR to RGB then normalizes to range 0:1
	input arg: path of image file
	return: numpy array
	'''
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, rs, cv2.INTER_LINEAR)
	image = image/255.0
	return image

def draw_outputs(img, outputs, class_names, i,color, fps):
	boxes, objectness, classes, nums = outputs
	boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
	wh = np.flip(img.shape[0:2])
	
	x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
	x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
	img = cv2.rectangle(img, x1y1, x2y2, color, 2)
	img = cv2.putText(img, '{} {:.4f}'.format(class_names[int(classes[i])], objectness[i]), x1y1, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
	img = cv2.putText(img, 'fps: {:.2f}'.format(fps), (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
			
	return img

def logging():
    console = logging.StreamHandler()
    logging.getLogger().setLevel(logging.NOTSET)
    # consoleformatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    consoleformatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console.setFormatter(consoleformatter)
    logging.getLogger().addHandler(console)

    return logging.getLogger()

def open_labels(ladels_path):
	with open(ladels_path, 'r') as label_file:
		__labels = [l.strip() for l in label_file.readlines()]
	return __labels

def open_json(CFG):
	with open(CFG) as json_file:
		cfg = json.load(json_file)
	return cfg
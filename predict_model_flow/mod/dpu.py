# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from .yolov3 import yolo_boxes, yolo_nms
import tensorflow as tf
from tensorflow.keras.layers import (
	Lambda,
)

try:
	import xir
	import vart
except:
	pass


class XMODEL:
	def __init__(self, model, name):
		self.__m = model # model path
		self.subgraphs = list()
		self.runner = None
		self.name = name
		self.outputs = []

	def get_child_subgraph_dpu(self) -> list:
		graph = xir.Graph.deserialize(self.__m)
		assert graph is not None, "'graph' should not be None."

		root_subgraph = graph.get_root_subgraph()
		assert (root_subgraph is not None), "Failed to get root subgraph of input Graph object."

		if root_subgraph.is_leaf:
			return []

		child_subgraphs = root_subgraph.toposort_child_subgraph()
		assert child_subgraphs is not None and len(child_subgraphs) > 0

		self.subgraphs = [
			cs
			for cs in child_subgraphs
			if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
		]

		return self.subgraphs
	
	def get_dpu_runner(self, graph=None):
		if graph is None:
			self.runner = vart.Runner.create_runner(self.subgraphs[0], 'run')
		else:
			self.runner = vart.Runner.create_runner(graph[0], 'run')
		return self.runner
	
	def init(self):
		self.get_child_subgraph_dpu()
		self.get_dpu_runner()

	def predict(self, input_data, outputs):
		job_id = self.runner.execute_async(input_data, outputs)
		self.runner.wait(job_id)

		return outputs
	
	def get_input_tensors(self):
		if self.runner is None:
			return None
		else:
			return self.runner.get_input_tensors()
	
	def get_output_tensors(self):
		if self.runner is None:
			return None
		else:
			return self.runner.get_output_tensors()
	
class YOLOV3(XMODEL):
	def __init__(self, model, name, input_size, box_max_num, anchors, classed_num, iou, nms, conf):
		super().__init__(model, name)
		self.prediction = []
		self.anchors = anchors
		self.classes = classed_num
		self.iou_th = iou
		self.nms_th = nms
		self.conf_th = conf
		self.box_num = box_max_num
		self.input_size = input_size

	def sort_boxes(self):
		if len(self.outputs) == 3:
			''' yolo '''
			boxes = np.array([[0, 1, 2], [0, 0, 0]], dtype=int)
			boxes[1][0] = self.outputs[0].shape[1]
			boxes[1][1] = self.outputs[1].shape[1]
			boxes[1][2] = self.outputs[2].shape[1]

		elif len(self.outputs) == 2:
			''' tiny yolo '''
			boxes = np.array([[0, 1], [0, 0]], dtype=int)
			boxes[1][0] = self.outputs[0].shape[1]
			boxes[1][1] = self.outputs[1].shape[1]

		sorted = list(np.argsort(boxes[1:]))
		sorted = sorted[0]

		return sorted



	def pred_boxes(self):
		if len(self.outputs) == 3:
			''' yolo '''
			a64 = np.reshape(self.outputs[self.sorted[2]], (1, int(self.input_size/8), int(self.input_size/8), 3, 5+self.classes))
			a32 = np.reshape(self.outputs[self.sorted[1]], (1, int(self.input_size/16), int(self.input_size/16), 3, 5+self.classes))
			a16 = np.reshape(self.outputs[self.sorted[0]], (1, int(self.input_size/32), int(self.input_size/32), 3, 5+self.classes))

			b0 = Lambda(lambda x: yolo_boxes(x, self.anchors[:3], self.classes), name='yb0')(a64)
			b1 = Lambda(lambda x: yolo_boxes(x, self.anchors[3:6], self.classes), name='yb1')(a32)
			b2 = Lambda(lambda x: yolo_boxes(x, self.anchors[6:], self.classes), name='yb2')(a16)

			self.prediction = Lambda(
				lambda x: yolo_nms(x, self.classes, self.box_num, self.iou_th, self.nms_th),
				name='yolo_nms')((b0[:3], b1[:3], b2[:3]))

			boxes, scores, classes, nums = self.prediction
			return np.array(boxes), np.array(scores), np.array(classes), np.array(nums)

		elif len(self.outputs) == 2:
			''' tiny yolo '''
			o_13 = self.outputs[0]
			o_26 = self.outputs[1]

			a32 = np.reshape(o_13, (1, int(self.input_size/32), int(self.input_size/32), 3, 5+self.classes))
			a16 = np.reshape(o_26, (1, int(self.input_size/16), int(self.input_size/16), 3, 5+self.classes))

			b1 = Lambda(lambda x: yolo_boxes(x, self.anchors[3:6], self.classes), name='yb0')(a32)
			b2 = Lambda(lambda x: yolo_boxes(x, self.anchors[0:3], self.classes), name='yb1')(a16)

			self.prediction =  Lambda(
				lambda x: yolo_nms(x, self.classes, self.box_num, self.iou_th, self.nms_th),
				name='yolo_nms')((b1[:3], b2[:3]))

			boxes, scores, classes, nums = self.prediction
			return np.array(boxes), np.array(scores), np.array(classes), np.array(nums)

			# a32 = np.reshape(self.outputs[self.sorted[1]], (1, int(self.input_size/16), int(self.input_size/16), 3, 5+self.classes))
			# a16 = np.reshape(self.outputs[self.sorted[0]], (1, int(self.input_size/32), int(self.input_size/32), 3, 5+self.classes))

			# b0 = Lambda(lambda x: yolo_boxes(x, self.anchors[0:3], self.classes), name='yb1')(a32)
			# b1 = Lambda(lambda x: yolo_boxes(x, self.anchors[3:], self.classes), name='yb2')(a16)

			# self.prediction = Lambda(
			# 		lambda x: yolo_nms(x, self.classes, self.box_num, self.iou_th, self.nms_th),
			# 		name='yolo_nms')((b0[:3], b1[:3]))
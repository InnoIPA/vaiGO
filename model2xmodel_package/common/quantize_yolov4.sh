#!/bin/bash

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
# 	Hueiru, hueiru_chen@innodisk.com, innodick Inc
# 	Wilson, wilson_yeh@innodisk.com, innodick Inc


INPUT_SIZE=$1
INPUT_CHANNEL=$2
INPUT_NODE=$3
# INPUT_NODE="input_1"
OUTPUT_NODE=$4
# OUTPUT_NODE="conv2d_20/BiasAdd,conv2d_23/BiasAdd"

cp ../common/q.py ./

path="./train.txt"

lines=$(wc -l < "$path")
CALIB_ITER=$((count + lines))

echo "find $CALIB_ITER image path"

echo ${INPUT_SIZE} > model_input_size.txt
echo ${INPUT_NODE} > model_input_node_name.txt

run_quant() {

	vai_q_tensorflow --version

	# quantize
	vai_q_tensorflow quantize \
	--input_frozen_graph ./output.pb \
	--input_fn           q.calib_input \
	--output_dir         ./ \
	--input_nodes        ${INPUT_NODE} \
	--output_nodes       ${OUTPUT_NODE} \
	--input_shapes       ?,${INPUT_SIZE},${INPUT_SIZE},${INPUT_CHANNEL} \
	--calib_iter         ${CALIB_ITER} \
	--gpu                0
}

quant() {

	echo "-----------------------------------------"
	echo "QUANTIZE STARTED.."
	echo "-----------------------------------------"

	run_quant 2>&1 | tee
 
	echo "-----------------------------------------"
	echo "QUANTIZED COMPLETED"
	echo "-----------------------------------------"
}

quant
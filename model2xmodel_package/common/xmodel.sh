INPUT_W=$1
INPUT_H=$2
INPUT_CHANNEL=$3
ARCH_JSON=$4
NET_NAME=$5

vai_c_tensorflow --frozen_pb ./quantize_eval_model.pb \
                 --arch ${ARCH_JSON} \
		 --output_dir ./ \
		 --net_name ${NET_NAME} \
		 --options "{'mode':'normal','save_kernel':'', 'input_shape':'1,${INPUT_W},${INPUT_H},${INPUT_CHANNEL}'}"

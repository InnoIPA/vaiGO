INPUT_SIZE=$1
INPUT_CHANNEL=$2
ARCH_JSON=$3
NET_NAME=$4

vai_c_tensorflow --frozen_pb ./quantize_eval_model.pb \
                 --arch ${ARCH_JSON} \
		 --output_dir ./ \
		 --net_name ${NET_NAME} \
		 --options "{'mode':'normal','save_kernel':'', 'input_shape':'1,${INPUT_SIZE},${INPUT_SIZE},${INPUT_CHANNEL}'}"
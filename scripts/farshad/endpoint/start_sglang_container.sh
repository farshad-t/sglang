
IMAGE_NAME=${IMAGE_NAME:-sglang-worker-router:latest}
#IMAGE_NAME=intel/intel-optimized-pytorch:mlperf-inference-6.0-llama3.1_8b_cpu
CUR_DIR=${PWD}

MODEL_DIR=/data2/llama/
DATA_DIR=/data/tattafos/cnn_dailymail/validation-data/
MLPERF_DIR=/data/tattafos/mlperf-inference-v6.0/closed/Intel/code/llama3.1-8b/pytorch-cpu
LOG_DIR=${CUR_DIR}/logs-docker
mkdir -p ${LOG_DIR}

CONTAINER_NAME=${CONTAINER_NAME:-taf-sglang}

docker run --privileged -it \
	--shm-size=4g \
	--ipc=host \
	--network=host \
	-e http_proxy=${http_proxy} \
	-e https_proxy=${https_proxy} \
	-e no_proxy=${no_proxy} \
	-v ${DATA_DIR}:/data \
	-v ${MODEL_DIR}:/model \
	-v ${LOG_DIR}:/logs \
	-v ${CUR_DIR}:/code \
	-v ${MLPERF_DIR}:/workspace \
	--name ${CONTAINER_NAME} \
	--entrypoint /bin/bash \
	${IMAGE_NAME} 


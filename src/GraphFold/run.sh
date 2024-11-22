BIN=../../bin/pattern_gpu_GF_LUT
GRAPH=../../datasets/mico/graph
export CUDA_VISIBLE_DEVICES=0
${BIN} ${GRAPH} $1 $2

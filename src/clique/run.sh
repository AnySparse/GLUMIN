BIN=../../bin/clique_GM_LUT
GRAPH=../../datasets/mico/graph
export CUDA_VISIBLE_DEVICES=0
${BIN} ${GRAPH} $1

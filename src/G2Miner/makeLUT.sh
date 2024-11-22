#!/bin/bash
set -e

PATTERN="P1"

# cd gpu_GM_LUT_kernels
# cp -f ./GM_LUT_kernels/${PATTERN}.yml.cuh ./GM_LUT.cuh
# cp -f ./GM_kernels/${PATTERN}-GM.yml.cuh ./BS_edge.cuh
# cd ..
make clean
make

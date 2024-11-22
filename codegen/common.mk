DEBUG ?= 0
CUDA_HOME=/usr/local/cuda-12/
CUB_DIR=../thirdparty/cub
CC=gcc
CXX=g++
MPICC=mpicc
MPICXX=mpicxx
NVCC=nvcc
CUDA_ARCH := -gencode arch=compute_70,code=sm_70
CXXFLAGS=-Wall -fopenmp -std=c++11 -march=native -fopenmp -l/home/linzhiheng/thirdparty/thirdparty/openmpi/lib
ICPCFLAGS=-O3 -Wall -qopenmp
NVFLAGS=$(CUDA_ARCH)
#NVFLAGS+=-Xptxas -v
NVFLAGS+=-DUSE_GPU
NVFLAGS+=-Xcompiler -fopenmp


ifeq ($(VTUNE), 1)
	CXXFLAGS += -g
endif
ifeq ($(NVPROF), 1)
	NVFLAGS += -lineinfo
endif

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0
	NVFLAGS += -G
else
	CXXFLAGS += -O3
	NVFLAGS += -O3 -w 
endif

INCLUDES = -I./include


VPATH += ./src
OBJS=main.o VertexSet.o graph.o

# CUDA vertex parallel
ifneq ($(VPAR),)
NVFLAGS += -DVERTEX_PAR
endif

# CUDA CTA centric
ifneq ($(CTA),)
NVFLAGS += -DCTA_CENTRIC
endif

ifneq ($(PROFILE),)
CXXFLAGS += -DPROFILING
endif

ifneq ($(USE_SET_OPS),)
CXXFLAGS += -DUSE_MERGE
endif

ifneq ($(USE_SIMD),)
CXXFLAGS += -DSI=0
endif

# counting or listing
ifneq ($(COUNT),)
NVFLAGS += -DDO_COUNT
endif

# GPU vertex/edge parallel 
ifeq ($(VERTEX_PAR),)
NVFLAGS += -DEDGE_PAR
endif

# CUDA unified memory
ifneq ($(USE_UM),)
NVFLAGS += -DUSE_UM
endif

# kernel fission
ifneq ($(FISSION),)
NVFLAGS += -DFISSION
endif


include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./gpu_GF_LUT_kernels
all: pattern_gpu_GF_LUT

pattern_gpu_GF_LUT: $(OBJS) gpu_GF_LUT.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) gpu_GF_LUT.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels -I./gpu_kernels_split
all: clique_GF_LUT clique_GM clique_GM_LUT

clique_GM: $(OBJS) clique_GM.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) clique_GM.o -o $@ $(LIBS)
	mv $@ $(BIN)

clique_GM_LUT: $(OBJS) clique_GM_LUT.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) clique_GM_LUT.o -o $@ $(LIBS)
	mv $@ $(BIN)

clique_GF_LUT: $(OBJS) clique_GF_LUT.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) clique_GF_LUT.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm -f *.o

.PHONY: clean


include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels -I./gpu_kernels -I./gpu_LUT_kernels -I./gpu_GM_LUT_kernels
all: cpu_serials pattern_multigpu_GM_LUT

cpu_serials: $(OBJS) cpu_serials.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) cpu_serials.o $(OBJS) -o $@ -lgomp
	#mv $@ $(BIN)

pattern_multigpu_GM_LUT: $(OBJS) multigpu_GM_LUT.o scheduler.o
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) multigpu_GM_LUT.o scheduler.o -o $@ $(LIBS)
	mv $@ $(BIN)

clean:
	rm *.o

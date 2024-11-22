include common.mk
COMMON_INCLUDES=-I./include -I./include/generated 
INCLUDES=$(COMMON_INCLUDES) -I./include/gpu_kernels

all: $(OBJS)
	$(NVCC) $(NVFLAGS) $(INCLUDES) $(OBJS) -o gpm

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $<

%.o: %.cu
	$(NVCC) $(NVFLAGS) $(INCLUDES) -c $<

clean:
	rm *.o gpm

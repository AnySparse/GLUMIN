include ../common.mk
#INCLUDES+=-I$(CUB_DIR)
INCLUDES+=-I./cpu_kernels
all: automine_LUT 

automine_LUT: $(OBJS) cpu_serials.o
	$(CXX) $(CXXFLAGS) $(INCLUDES) cpu_serials.o $(OBJS) -o $@ -lgomp
	mv $@ $(BIN)

clean:
	rm *.o

include ../../common/make.config

#C compiler
CC = g++

SRC = gaussian.cu.out.cc

CC_FLAGS = -g -O3

EXE = gaussian

release:
	$(CC) $(KERNEL_DIM) $(CC_FLAGS) -lOpenCL  -fopenmp $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -L /home/shichangqing/nkocl/lib -I /home/shichangqing/nkocl/include -lnkocl
	
clean: 
	rm -f $(EXE) 

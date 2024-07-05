include ../../common/make.config

#C compiler
CC = g++

CC_FLAGS = -g -O3 -Wall

SRC = needle.cu.out.cc needle_kernel.cu.out.h

EXE = nw

release: $(SRC)
	$(CC) ${KERNEL_DIM} $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -L /home/shichangqing/nkocl/lib -I /home/shichangqing/nkocl/include -lOpenCL -lnkocl

clean:
	rm -f $(EXE)

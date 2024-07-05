include ../../common/make.config
#OPENCL_DIR = /af21/den4gr/NVIDIA_GPU_Computing_SDK/

# C compiler
CC = g++
CC_FLAGS = -O2 
#INCLUDES = -I$(OPENCL_DIR)/OpenCL/common/inc -I$(OPENCL_DIR)/shared/inc/
INCLUDES = -I$(OPENCL_INC)

all: naive

naive: ex_particle_CUDA_naive_seq.cu.out.cc
	$(CC) -I$(OPENCL_INC) -L$(OPENCL_LIB) -g -lm -O3 ex_particle_CUDA_naive_seq.cu.out.cc -o particlefilter_naive -lOpenCL -L /home/shichangqing/nkocl/lib -I /home/shichangqing/nkocl/include -lnkocl
	

clean:
	rm particlefilter_naive

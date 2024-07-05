




include ../../common/make.config

CC := g++

SRC = hotspot.cu.out.cc

EXE = hotspot

release: $(SRC)
	$(CC) $(KERNEL_DIM) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -L /home/shichangqing/nkocl/lib -I /home/shichangqing/nkocl/include -lOpenCL -lnkocl

enum: $(SRC)
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) 

debug: $(SRC)
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) 

debugenum: $(SRC)
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) 

clean: $(SRC)
	rm -f $(EXE) $(EXE).linkinfo result.txt


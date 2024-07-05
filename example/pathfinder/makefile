



include ../../common/make.config

CC := g++

SRC = pathfinder.cu.out.cc

EXE = pathfinder

release:
	$(CC) $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) -L /home/shichangqing/nkocl/lib -I /home/shichangqing/nkocl/include -lnkocl

enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$$(OPENCL_LIB) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$$(OPENCL_LIB) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(OPENCL_INC) -L$(OPENCL_LIB) 

clean:
	rm -f pathfinder

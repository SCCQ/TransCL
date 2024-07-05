#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include "nkapi.h"
#include <sys/time.h>
#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1 // halo width along one direction when advancing to the next iteration

#define BENCH_PRINT

void run(int argc, char** argv);

int rows, cols;
int* data;
int** wall;
int* result;
#define M_SEED 9
int pyramid_height;

//#define BENCH_PRINT

unsigned int totalKernelTime = 0;
void
init(int argc, char** argv)
{
	if(argc==4){

		cols = atoi(argv[1]);

		rows = atoi(argv[2]);

                pyramid_height=atoi(argv[3]);
	}else{
                printf("Usage: dynproc row_len col_len pyramid_height\n");
                exit(0);
        }
	data = new int[rows*cols];

	wall = new int*[rows];

	for(int n=0; n<rows; n++)

		wall[n]=data+cols*n;

	result = new int[cols];

	

	int seed = M_SEED;

	srand(seed);



	for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            wall[i][j] = rand() % 10;

        }

    }

#ifdef BENCH_PRINT

    for (int i = 0; i < rows; i++)

    {

        for (int j = 0; j < cols; j++)

        {

            printf("%d ",wall[i][j]) ;

        }

        printf("\n") ;

    }

#endif
}

void 
fatal(char *s)
{
	fprintf(stderr, "error: %s\n", s);

}

#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

 ocl_kernel_prop dynproc_kernel()
{
ocl_kernel_prop prop;
prop.src = R"NKT(#include "functions.h"
#define IN_RANGE(x, min, max)   ((x)>=(min) && (x)<=(max))
#define CLAMP_RANGE(x, min, max) x = (x<(min)) ? min : ((x>(max)) ? max : x )
#define MIN(a, b) ((a)<=(b) ? (a) : (b))
#define BLOCK_SIZE 256
#define STR_SIZE 256
#define DEVICE 0
#define HALO 1
__kernel void dynproc_kernel(
                int iteration,
                __global int *gpuWall,
                __global int *gpuSrc,
                __global int *gpuResults,
                int cols,
                int rows,
                int startStep,
                int border)
{

        __local int prev[BLOCK_SIZE];
        __local int result[BLOCK_SIZE];

 int bx = get_group_id(0);
 int tx=get_local_id(0);

        // each block finally computes result for a small block
        // after N iterations. 
        // it is the non-overlapping small blocks that cover 
        // all the input data

        // calculate the small block size
 int small_block_cols = BLOCK_SIZE-iteration*HALO*2;

        // calculate the boundary for the block according to 
        // the boundary of its small block
        int blkX = small_block_cols*bx-border;
        int blkXmax = blkX+BLOCK_SIZE-1;

        // calculate the global thread coordination
 int xidx = blkX+tx;

        // effective range within this block that falls within 
        // the valid range of the input data
        // used to rule out computation outside the boundary.
        int validXmin = (blkX < 0) ? -blkX : 0;
        int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

        int W = tx-1;
        int E = tx+1;

        W = (W < validXmin) ? validXmin : W;
        E = (E > validXmax) ? validXmax : E;

        bool isValid = IN_RANGE(tx, validXmin, validXmax);

 if(IN_RANGE(xidx, 0, cols-1)){
            prev[tx] = gpuSrc[xidx];
 }
 barrier(CLK_LOCAL_MEM_FENCE); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
        bool computed;
        for (int i=0; i<iteration ; i++){
            computed = false;
            if( IN_RANGE(tx, i+1, BLOCK_SIZE-i-2) &&
                  isValid){
                  computed = true;
                  int left = prev[W];
                  int up = prev[tx];
                  int right = prev[E];
                  int shortest = MIN(left, up);
                  shortest = MIN(shortest, right);
                  int index = cols*(startStep+i)+xidx;
                  result[tx] = shortest + gpuWall[index];

            }
            barrier(CLK_LOCAL_MEM_FENCE);
            if(i==iteration-1)
                break;
            if(computed)  //Assign the computation range
                prev[tx]= result[tx];
     barrier(CLK_LOCAL_MEM_FENCE); // [Ronny] Added sync to avoid race on prev Aug. 14 2012
      }

      // update the global memory
      // after the last iteration, only threads coordinated within the 
      // small block perform the calculation and switch on ``computed''
      if (computed){
          gpuResults[xidx]=result[tx];
      }
})NKT";
prop.kernel_name = "dynproc_kernel";
prop.args_num = 8;
prop.args_type.push_back(NKOCL_INT);
prop.args_type.push_back(NKOCL_CLBUFFER);
prop.args_type.push_back(NKOCL_CLBUFFER);
prop.args_type.push_back(NKOCL_CLBUFFER);
prop.args_type.push_back(NKOCL_INT);
prop.args_type.push_back(NKOCL_INT);
prop.args_type.push_back(NKOCL_INT);
prop.args_type.push_back(NKOCL_INT);

return prop; 
 }

/*
   compute N time steps
*/
int calc_path(CUdeviceptr *hgpuWall, CUdeviceptr *hgpuResult[2], int rows, int cols,
	 int pyramid_height, int blockCols, int borderCols, int size)
{
        dim3 dimBlock(BLOCK_SIZE);
        dim3 dimGrid(blockCols);  
	
        int src = 1, dst = 0;
        struct timeval time_start;
        gettimeofday(&time_start, NULL);
        ocl_kernel_prop prop = dynproc_kernel();
	for (int t = 0; t < rows-1; t+=pyramid_height) {
            int temp = src;
            src = dst;
            dst = temp;
            int a = MIN(pyramid_height, rows-t-1);
            std::pair<void*, unsigned long long> fuc1((void*)hgpuWall, sizeof(int)*(size-cols));
            std::pair<void*, unsigned long long> fuc2((void*)hgpuResult[src], sizeof(int)*cols);
            std::pair<void*, unsigned long long> fuc3((void*)hgpuResult[dst], sizeof(int)*cols);
            void *param[] = {(void*)&a, (void*)&fuc1, (void*)&fuc2, (void*)&fuc3, (void*)&cols, (void*)&rows, (void*)&t, (void*)&borderCols};
            cuLaunchKernel((void*)&prop, dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, 0, 0, param, 0);
	}
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
        return dst;
}

int main(int argc, char** argv)
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 1) cudaSetDevice(DEVICE);
    struct timeval time_start;
    gettimeofday(&time_start, NULL);
    run(argc,argv);
    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
    printf("\nTime total (including memory transfers)\t%f sec\n", time_total * 1e-6);
    printf("Time for OPENCL kernels:\t%f sec\n",totalKernelTime * 1e-6);
    return EXIT_SUCCESS;
}

void run(int argc, char** argv)
{
    init(argc, argv);

    /* --------------- pyramid parameters --------------- */
    int borderCols = (pyramid_height)*HALO;
    int smallBlockCol = BLOCK_SIZE-(pyramid_height)*HALO*2;
    int blockCols = cols/smallBlockCol+((cols%smallBlockCol==0)?0:1);

    printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
	pyramid_height, cols, borderCols, BLOCK_SIZE, blockCols, smallBlockCol);
	
    CUdeviceptr *hgpuWall, *hgpuResult[2];
    int size = rows*cols;

    cudaMalloc((void **)&hgpuResult[0], sizeof(int)*cols);
    cudaMalloc((void **)&hgpuResult[1], sizeof(int)*cols);
    cudaMemcpy(hgpuResult[0], data, sizeof(int)*cols, cudaMemcpyHostToDevice);
    cudaMalloc((void**)&hgpuWall, sizeof(int)*(size-cols));
    cudaMemcpy(hgpuWall, data+cols, sizeof(int)*(size-cols), cudaMemcpyHostToDevice);


    int final_ret = calc_path(hgpuWall, hgpuResult, rows, cols,
	 pyramid_height, blockCols, borderCols, size);

    cudaMemcpy(result, hgpuResult[final_ret], sizeof(int)*cols, cudaMemcpyDeviceToHost);


#ifdef BENCH_PRINT

    for (int i = 0; i < cols; i++)

            printf("%d ",data[i]) ;

    printf("\n") ;

    for (int i = 0; i < cols; i++)

            printf("%d ",result[i]) ;

    printf("\n") ;

#endif


    cudaFree(hgpuWall);
    cudaFree(hgpuResult[0]);
    cudaFree(hgpuResult[1]);

    delete [] data;
    delete [] wall;
    delete [] result;

}


#include <stdio.h>
#include "nkapi.h"

#ifdef RD_WG_SIZE_0_0
	#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
	#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
	#define BLOCK_SIZE RD_WG_SIZE
#else
	#define BLOCK_SIZE 16
#endif

ocl_kernel_prop needle_cuda_shared_1()
{
    ocl_kernel_prop prop;
    prop.src = R"NKT(#include "functions.h"
    #define BLOCK_SIZE nktarg_1
    int maximumm(int a, int b, int c){
        int k;
        if( a <= b )
            k = b;
        else 
            k = a;
        if( k <=c )
            return(c);
        else
            return(k);

    }


    __kernel void needle_cuda_shared_1(__global int* referrence,
        __global int* matrix_cuda,
        int cols,
        int penalty,
        int i,
        int block_width)
    {
        int bx = get_group_id(0);
        int tx = get_local_id(0);

        int b_index_x = bx;
        int b_index_y = i - 1 - bx;

        int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
        int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
        int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
        int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

        __local int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
        __local int ref[BLOCK_SIZE][BLOCK_SIZE];

        if (tx == 0)
            temp[tx][0] = matrix_cuda[index_nw];


        for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
            ref[ty][tx] = referrence[index + cols * ty];

        barrier(CLK_LOCAL_MEM_FENCE);

        temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

        barrier(CLK_LOCAL_MEM_FENCE);

        temp[0][tx + 1] = matrix_cuda[index_n];

        barrier(CLK_LOCAL_MEM_FENCE);


        for( int m = 0 ; m < BLOCK_SIZE ; m++){
            if ( tx <= m ){
                int t_index_x =  tx + 1;
                int t_index_y =  m - tx + 1;

                temp[t_index_y][t_index_x] = maximumm(temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                    temp[t_index_y-1][t_index_x]  - penalty);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

            if ( tx <= m){
                int t_index_x =  tx + BLOCK_SIZE - m ;
                int t_index_y =  BLOCK_SIZE - tx;
                temp[t_index_y][t_index_x] = maximumm( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                    temp[t_index_y-1][t_index_x]  - penalty);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
            matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

    })NKT";
    prop.kernel_name = "needle_cuda_shared_1";
    prop.src = prop.src.replace(prop.src.find("nktarg_1"), 8, std::to_string(BLOCK_SIZE));
    prop.args_num = 6;
    prop.args_type.push_back(NKOCL_CLBUFFER);
    prop.args_type.push_back(NKOCL_CLBUFFER);
    prop.args_type.push_back(NKOCL_INT);
    prop.args_type.push_back(NKOCL_INT);
    prop.args_type.push_back(NKOCL_INT);
    prop.args_type.push_back(NKOCL_INT);
    return prop; 
}


ocl_kernel_prop needle_cuda_shared_2()
{
    ocl_kernel_prop prop;
    prop.src = R"NKT(#include "functions.h"
    #define BLOCK_SIZE nktarg_1

    int maximumm(int a, int b, int c){
        int k;
        if( a <= b )
            k = b;
        else 
            k = a;
        if( k <=c )
            return(c);
        else
            return(k);

    }
    
    __kernel void
    needle_cuda_shared_2(  __global int* referrence,
        __global int* matrix_cuda,

        int cols,
        int penalty,
        int i,
        int block_width)
    {

        int bx = get_group_id(0);
        int tx = get_local_id(0);

        int b_index_x = bx + block_width - i  ;
        int b_index_y = block_width - bx -1;

        int index   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( cols + 1 );
        int index_n   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + tx + ( 1 );
        int index_w   = cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x + ( cols );
        int index_nw =  cols * BLOCK_SIZE * b_index_y + BLOCK_SIZE * b_index_x;

        __local  int temp[BLOCK_SIZE+1][BLOCK_SIZE+1];
        __local  int ref[BLOCK_SIZE][BLOCK_SIZE];

        for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
            ref[ty][tx] = referrence[index + cols * ty];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (tx == 0)
            temp[tx][0] = matrix_cuda[index_nw];


        temp[tx + 1][0] = matrix_cuda[index_w + cols * tx];

        barrier(CLK_LOCAL_MEM_FENCE);

        temp[0][tx + 1] = matrix_cuda[index_n];

        barrier(CLK_LOCAL_MEM_FENCE);


        for( int m = 0 ; m < BLOCK_SIZE ; m++){

            if ( tx <= m ){
                int t_index_x =  tx + 1;
                int t_index_y =  m - tx + 1;

                temp[t_index_y][t_index_x] = maximumm( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                    temp[t_index_y-1][t_index_x]  - penalty);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for( int m = BLOCK_SIZE - 2 ; m >=0 ; m--){

            if ( tx <= m){
                int t_index_x =  tx + BLOCK_SIZE - m ;
                int t_index_y =  BLOCK_SIZE - tx;
                temp[t_index_y][t_index_x] = maximumm( temp[t_index_y-1][t_index_x-1] + ref[t_index_y-1][t_index_x-1],
                                                temp[t_index_y][t_index_x-1]  - penalty,
                    temp[t_index_y-1][t_index_x]  - penalty);
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }


        for ( int ty = 0 ; ty < BLOCK_SIZE ; ty++)
            matrix_cuda[index + ty * cols] = temp[ty+1][tx+1];

    })NKT";
    prop.kernel_name = "needle_cuda_shared_2";
    prop.src = prop.src.replace(prop.src.find("nktarg_1"), 8, std::to_string(BLOCK_SIZE));
    prop.args_num = 6;
    prop.args_type.push_back(NKOCL_CLBUFFER);
    prop.args_type.push_back(NKOCL_CLBUFFER);
    prop.args_type.push_back(NKOCL_INT);
    prop.args_type.push_back(NKOCL_INT);
    prop.args_type.push_back(NKOCL_INT);
    prop.args_type.push_back(NKOCL_INT);

    return prop; 
}


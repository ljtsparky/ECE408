#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define BLOCK_SIZE 16 //@@ You can change this

__constant__ float Constant_mask[3136];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K) //const int layer
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    const int shared_mem_size = BLOCK_SIZE + K -1;
    extern __shared__ float shared_mem[];
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Constant_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define shared_mem_3d(i2,i1,i0) shared_mem[ (i2) * (shared_mem_size*shared_mem_size) + (i1)*shared_mem_size + i0]
    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int blocknum_per_row = (Width_out - 1)/BLOCK_SIZE + 1; // how many features there are
    int w_out = BLOCK_SIZE * (by % blocknum_per_row) + tx; // width out for this thread would depend on remain 
    int h_out = BLOCK_SIZE * (by / blocknum_per_row) + ty; // height for this thread would depend on how many rows we've come through
    int batch_out = bz; //bz is for batches
    int feature_out = bx; //bx is for different output features

    int block_lu_x = BLOCK_SIZE * (by % blocknum_per_row);
    int block_lu_y = BLOCK_SIZE * (by / blocknum_per_row);
    if (Map_out == 4){
        for (int c = 0; c < Channel; c++){
            for(int i = ty; i < shared_mem_size; i += BLOCK_SIZE){
                for(int j = tx; j < shared_mem_size; j += BLOCK_SIZE){
                    if (block_lu_y + i < Height && block_lu_x + j < Width){
                        shared_mem_3d(c, i, j) = in_4d(batch_out, c, block_lu_y + i, block_lu_x + j);
                    }
                    
                }
            }
        }
        __syncthreads();
    
        if (h_out < Height_out && w_out < Width_out){
            float result = 0;
            result += shared_mem_3d(0, ty + 0, tx + 0) * mask_4d(feature_out, 0, 0, 0)
                    + shared_mem_3d(0, ty + 0, tx + 1) * mask_4d(feature_out, 0, 0, 1)
                    + shared_mem_3d(0, ty + 0, tx + 2) * mask_4d(feature_out, 0, 0, 2)
                    + shared_mem_3d(0, ty + 0, tx + 3) * mask_4d(feature_out, 0, 0, 3)
                    + shared_mem_3d(0, ty + 0, tx + 4) * mask_4d(feature_out, 0, 0, 4)
                    + shared_mem_3d(0, ty + 0, tx + 5) * mask_4d(feature_out, 0, 0, 5)
                    + shared_mem_3d(0, ty + 0, tx + 6) * mask_4d(feature_out, 0, 0, 6)
                    + shared_mem_3d(0, ty + 1, tx + 0) * mask_4d(feature_out, 0, 1, 0)
                    + shared_mem_3d(0, ty + 1, tx + 1) * mask_4d(feature_out, 0, 1, 1)
                    + shared_mem_3d(0, ty + 1, tx + 2) * mask_4d(feature_out, 0, 1, 2)
                    + shared_mem_3d(0, ty + 1, tx + 3) * mask_4d(feature_out, 0, 1, 3)
                    + shared_mem_3d(0, ty + 1, tx + 4) * mask_4d(feature_out, 0, 1, 4)
                    + shared_mem_3d(0, ty + 1, tx + 5) * mask_4d(feature_out, 0, 1, 5)
                    + shared_mem_3d(0, ty + 1, tx + 6) * mask_4d(feature_out, 0, 1, 6);
            result += shared_mem_3d(0, ty + 2, tx + 0) * mask_4d(feature_out, 0, 2, 0)
                    + shared_mem_3d(0, ty + 2, tx + 1) * mask_4d(feature_out, 0, 2, 1)
                    + shared_mem_3d(0, ty + 2, tx + 2) * mask_4d(feature_out, 0, 2, 2)
                    + shared_mem_3d(0, ty + 2, tx + 3) * mask_4d(feature_out, 0, 2, 3)
                    + shared_mem_3d(0, ty + 2, tx + 4) * mask_4d(feature_out, 0, 2, 4)
                    + shared_mem_3d(0, ty + 2, tx + 5) * mask_4d(feature_out, 0, 2, 5)
                    + shared_mem_3d(0, ty + 2, tx + 6) * mask_4d(feature_out, 0, 2, 6);
            result += shared_mem_3d(0, ty + 3, tx + 0) * mask_4d(feature_out, 0, 3, 0)
                    + shared_mem_3d(0, ty + 3, tx + 1) * mask_4d(feature_out, 0, 3, 1)
                    + shared_mem_3d(0, ty + 3, tx + 2) * mask_4d(feature_out, 0, 3, 2)
                    + shared_mem_3d(0, ty + 3, tx + 3) * mask_4d(feature_out, 0, 3, 3)
                    + shared_mem_3d(0, ty + 3, tx + 4) * mask_4d(feature_out, 0, 3, 4)
                    + shared_mem_3d(0, ty + 3, tx + 5) * mask_4d(feature_out, 0, 3, 5)
                    + shared_mem_3d(0, ty + 3, tx + 6) * mask_4d(feature_out, 0, 3, 6)
                    + shared_mem_3d(0, ty + 4, tx + 0) * mask_4d(feature_out, 0, 4, 0)
                    + shared_mem_3d(0, ty + 4, tx + 1) * mask_4d(feature_out, 0, 4, 1)
                    + shared_mem_3d(0, ty + 4, tx + 2) * mask_4d(feature_out, 0, 4, 2)
                    + shared_mem_3d(0, ty + 4, tx + 3) * mask_4d(feature_out, 0, 4, 3)
                    + shared_mem_3d(0, ty + 4, tx + 4) * mask_4d(feature_out, 0, 4, 4)
                    + shared_mem_3d(0, ty + 4, tx + 5) * mask_4d(feature_out, 0, 4, 5)
                    + shared_mem_3d(0, ty + 4, tx + 6) * mask_4d(feature_out, 0, 4, 6);
            result += shared_mem_3d(0, ty + 5, tx + 0) * mask_4d(feature_out, 0, 5, 0)
                    + shared_mem_3d(0, ty + 5, tx + 1) * mask_4d(feature_out, 0, 5, 1)
                    + shared_mem_3d(0, ty + 5, tx + 2) * mask_4d(feature_out, 0, 5, 2)
                    + shared_mem_3d(0, ty + 5, tx + 3) * mask_4d(feature_out, 0, 5, 3)
                    + shared_mem_3d(0, ty + 5, tx + 4) * mask_4d(feature_out, 0, 5, 4)
                    + shared_mem_3d(0, ty + 5, tx + 5) * mask_4d(feature_out, 0, 5, 5)
                    + shared_mem_3d(0, ty + 5, tx + 6) * mask_4d(feature_out, 0, 5, 6)
                    + shared_mem_3d(0, ty + 6, tx + 0) * mask_4d(feature_out, 0, 6, 0)
                    + shared_mem_3d(0, ty + 6, tx + 1) * mask_4d(feature_out, 0, 6, 1)
                    + shared_mem_3d(0, ty + 6, tx + 2) * mask_4d(feature_out, 0, 6, 2)
                    + shared_mem_3d(0, ty + 6, tx + 3) * mask_4d(feature_out, 0, 6, 3)
                    + shared_mem_3d(0, ty + 6, tx + 4) * mask_4d(feature_out, 0, 6, 4)
                    + shared_mem_3d(0, ty + 6, tx + 5) * mask_4d(feature_out, 0, 6, 5)
                    + shared_mem_3d(0, ty + 6, tx + 6) * mask_4d(feature_out, 0, 6, 6);
            out_4d(batch_out, feature_out, h_out, w_out) = result;
        }
    }
    else{
        for (int c = 0; c < Channel; c++){
            for(int i = ty; i < shared_mem_size; i += BLOCK_SIZE){
                for(int j = tx; j < shared_mem_size; j += BLOCK_SIZE){
                    if (block_lu_y + i < Height && block_lu_x + j < Width){
                        shared_mem_3d(c, i, j) = in_4d(batch_out, c, block_lu_y + i, block_lu_x + j);
                    }
                    
                }
            }
        }
        __syncthreads();
    
        if (h_out < Height_out && w_out < Width_out){
            float result = 0;
            result += shared_mem_3d(0, ty + 0, tx + 0) * mask_4d(feature_out, 0, 0, 0)
                    + shared_mem_3d(0, ty + 0, tx + 1) * mask_4d(feature_out, 0, 0, 1)
                    + shared_mem_3d(0, ty + 0, tx + 2) * mask_4d(feature_out, 0, 0, 2)
                    + shared_mem_3d(0, ty + 0, tx + 3) * mask_4d(feature_out, 0, 0, 3)
                    + shared_mem_3d(0, ty + 0, tx + 4) * mask_4d(feature_out, 0, 0, 4)
                    + shared_mem_3d(0, ty + 0, tx + 5) * mask_4d(feature_out, 0, 0, 5)
                    + shared_mem_3d(0, ty + 0, tx + 6) * mask_4d(feature_out, 0, 0, 6)
                    + shared_mem_3d(0, ty + 1, tx + 0) * mask_4d(feature_out, 0, 1, 0)
                    + shared_mem_3d(0, ty + 1, tx + 1) * mask_4d(feature_out, 0, 1, 1)
                    + shared_mem_3d(0, ty + 1, tx + 2) * mask_4d(feature_out, 0, 1, 2)
                    + shared_mem_3d(0, ty + 1, tx + 3) * mask_4d(feature_out, 0, 1, 3)
                    + shared_mem_3d(0, ty + 1, tx + 4) * mask_4d(feature_out, 0, 1, 4)
                    + shared_mem_3d(0, ty + 1, tx + 5) * mask_4d(feature_out, 0, 1, 5)
                    + shared_mem_3d(0, ty + 1, tx + 6) * mask_4d(feature_out, 0, 1, 6);
            result += shared_mem_3d(0, ty + 2, tx + 0) * mask_4d(feature_out, 0, 2, 0)
                    + shared_mem_3d(0, ty + 2, tx + 1) * mask_4d(feature_out, 0, 2, 1)
                    + shared_mem_3d(0, ty + 2, tx + 2) * mask_4d(feature_out, 0, 2, 2)
                    + shared_mem_3d(0, ty + 2, tx + 3) * mask_4d(feature_out, 0, 2, 3)
                    + shared_mem_3d(0, ty + 2, tx + 4) * mask_4d(feature_out, 0, 2, 4)
                    + shared_mem_3d(0, ty + 2, tx + 5) * mask_4d(feature_out, 0, 2, 5)
                    + shared_mem_3d(0, ty + 2, tx + 6) * mask_4d(feature_out, 0, 2, 6);
            result += shared_mem_3d(0, ty + 3, tx + 0) * mask_4d(feature_out, 0, 3, 0)
                    + shared_mem_3d(0, ty + 3, tx + 1) * mask_4d(feature_out, 0, 3, 1)
                    + shared_mem_3d(0, ty + 3, tx + 2) * mask_4d(feature_out, 0, 3, 2)
                    + shared_mem_3d(0, ty + 3, tx + 3) * mask_4d(feature_out, 0, 3, 3)
                    + shared_mem_3d(0, ty + 3, tx + 4) * mask_4d(feature_out, 0, 3, 4)
                    + shared_mem_3d(0, ty + 3, tx + 5) * mask_4d(feature_out, 0, 3, 5)
                    + shared_mem_3d(0, ty + 3, tx + 6) * mask_4d(feature_out, 0, 3, 6)
                    + shared_mem_3d(0, ty + 4, tx + 0) * mask_4d(feature_out, 0, 4, 0)
                    + shared_mem_3d(0, ty + 4, tx + 1) * mask_4d(feature_out, 0, 4, 1)
                    + shared_mem_3d(0, ty + 4, tx + 2) * mask_4d(feature_out, 0, 4, 2)
                    + shared_mem_3d(0, ty + 4, tx + 3) * mask_4d(feature_out, 0, 4, 3)
                    + shared_mem_3d(0, ty + 4, tx + 4) * mask_4d(feature_out, 0, 4, 4)
                    + shared_mem_3d(0, ty + 4, tx + 5) * mask_4d(feature_out, 0, 4, 5)
                    + shared_mem_3d(0, ty + 4, tx + 6) * mask_4d(feature_out, 0, 4, 6);
            result += shared_mem_3d(0, ty + 5, tx + 0) * mask_4d(feature_out, 0, 5, 0)
                    + shared_mem_3d(0, ty + 5, tx + 1) * mask_4d(feature_out, 0, 5, 1)
                    + shared_mem_3d(0, ty + 5, tx + 2) * mask_4d(feature_out, 0, 5, 2)
                    + shared_mem_3d(0, ty + 5, tx + 3) * mask_4d(feature_out, 0, 5, 3)
                    + shared_mem_3d(0, ty + 5, tx + 4) * mask_4d(feature_out, 0, 5, 4)
                    + shared_mem_3d(0, ty + 5, tx + 5) * mask_4d(feature_out, 0, 5, 5)
                    + shared_mem_3d(0, ty + 5, tx + 6) * mask_4d(feature_out, 0, 5, 6)
                    + shared_mem_3d(0, ty + 6, tx + 0) * mask_4d(feature_out, 0, 6, 0)
                    + shared_mem_3d(0, ty + 6, tx + 1) * mask_4d(feature_out, 0, 6, 1)
                    + shared_mem_3d(0, ty + 6, tx + 2) * mask_4d(feature_out, 0, 6, 2)
                    + shared_mem_3d(0, ty + 6, tx + 3) * mask_4d(feature_out, 0, 6, 3)
                    + shared_mem_3d(0, ty + 6, tx + 4) * mask_4d(feature_out, 0, 6, 4)
                    + shared_mem_3d(0, ty + 6, tx + 5) * mask_4d(feature_out, 0, 6, 5)
                    + shared_mem_3d(0, ty + 6, tx + 6) * mask_4d(feature_out, 0, 6, 6);
            result += shared_mem_3d(1, ty + 0, tx + 0) * mask_4d(feature_out, 1, 0, 0)
                    + shared_mem_3d(1, ty + 0, tx + 1) * mask_4d(feature_out, 1, 0, 1)
                    + shared_mem_3d(1, ty + 0, tx + 2) * mask_4d(feature_out, 1, 0, 2)
                    + shared_mem_3d(1, ty + 0, tx + 3) * mask_4d(feature_out, 1, 0, 3)
                    + shared_mem_3d(1, ty + 0, tx + 4) * mask_4d(feature_out, 1, 0, 4)
                    + shared_mem_3d(1, ty + 0, tx + 5) * mask_4d(feature_out, 1, 0, 5)
                    + shared_mem_3d(1, ty + 0, tx + 6) * mask_4d(feature_out, 1, 0, 6)
                    + shared_mem_3d(1, ty + 1, tx + 0) * mask_4d(feature_out, 1, 1, 0)
                    + shared_mem_3d(1, ty + 1, tx + 1) * mask_4d(feature_out, 1, 1, 1)
                    + shared_mem_3d(1, ty + 1, tx + 2) * mask_4d(feature_out, 1, 1, 2)
                    + shared_mem_3d(1, ty + 1, tx + 3) * mask_4d(feature_out, 1, 1, 3)
                    + shared_mem_3d(1, ty + 1, tx + 4) * mask_4d(feature_out, 1, 1, 4)
                    + shared_mem_3d(1, ty + 1, tx + 5) * mask_4d(feature_out, 1, 1, 5)
                    + shared_mem_3d(1, ty + 1, tx + 6) * mask_4d(feature_out, 1, 1, 6);
            result += shared_mem_3d(1, ty + 2, tx + 0) * mask_4d(feature_out, 1, 2, 0)
                    + shared_mem_3d(1, ty + 2, tx + 1) * mask_4d(feature_out, 1, 2, 1)
                    + shared_mem_3d(1, ty + 2, tx + 2) * mask_4d(feature_out, 1, 2, 2)
                    + shared_mem_3d(1, ty + 2, tx + 3) * mask_4d(feature_out, 1, 2, 3)
                    + shared_mem_3d(1, ty + 2, tx + 4) * mask_4d(feature_out, 1, 2, 4)
                    + shared_mem_3d(1, ty + 2, tx + 5) * mask_4d(feature_out, 1, 2, 5)
                    + shared_mem_3d(1, ty + 2, tx + 6) * mask_4d(feature_out, 1, 2, 6)
                    + shared_mem_3d(1, ty + 3, tx + 0) * mask_4d(feature_out, 1, 3, 0)
                    + shared_mem_3d(1, ty + 3, tx + 1) * mask_4d(feature_out, 1, 3, 1)
                    + shared_mem_3d(1, ty + 3, tx + 2) * mask_4d(feature_out, 1, 3, 2)
                    + shared_mem_3d(1, ty + 3, tx + 3) * mask_4d(feature_out, 1, 3, 3)
                    + shared_mem_3d(1, ty + 3, tx + 4) * mask_4d(feature_out, 1, 3, 4)
                    + shared_mem_3d(1, ty + 3, tx + 5) * mask_4d(feature_out, 1, 3, 5)
                    + shared_mem_3d(1, ty + 3, tx + 6) * mask_4d(feature_out, 1, 3, 6);
            result += shared_mem_3d(2, ty + 0, tx + 0) * mask_4d(feature_out, 2, 0, 0)
                    + shared_mem_3d(2, ty + 0, tx + 1) * mask_4d(feature_out, 2, 0, 1)
                    + shared_mem_3d(2, ty + 0, tx + 2) * mask_4d(feature_out, 2, 0, 2)
                    + shared_mem_3d(2, ty + 0, tx + 3) * mask_4d(feature_out, 2, 0, 3)
                    + shared_mem_3d(2, ty + 0, tx + 4) * mask_4d(feature_out, 2, 0, 4)
                    + shared_mem_3d(2, ty + 0, tx + 5) * mask_4d(feature_out, 2, 0, 5)
                    + shared_mem_3d(2, ty + 0, tx + 6) * mask_4d(feature_out, 2, 0, 6)
                    + shared_mem_3d(2, ty + 1, tx + 0) * mask_4d(feature_out, 2, 1, 0)
                    + shared_mem_3d(2, ty + 1, tx + 1) * mask_4d(feature_out, 2, 1, 1)
                    + shared_mem_3d(2, ty + 1, tx + 2) * mask_4d(feature_out, 2, 1, 2)
                    + shared_mem_3d(2, ty + 1, tx + 3) * mask_4d(feature_out, 2, 1, 3)
                    + shared_mem_3d(2, ty + 1, tx + 4) * mask_4d(feature_out, 2, 1, 4)
                    + shared_mem_3d(2, ty + 1, tx + 5) * mask_4d(feature_out, 2, 1, 5)
                    + shared_mem_3d(2, ty + 1, tx + 6) * mask_4d(feature_out, 2, 1, 6);
            result += shared_mem_3d(2, ty + 2, tx + 0) * mask_4d(feature_out, 2, 2, 0)
                    + shared_mem_3d(2, ty + 2, tx + 1) * mask_4d(feature_out, 2, 2, 1)
                    + shared_mem_3d(2, ty + 2, tx + 2) * mask_4d(feature_out, 2, 2, 2)
                    + shared_mem_3d(2, ty + 2, tx + 3) * mask_4d(feature_out, 2, 2, 3)
                    + shared_mem_3d(2, ty + 2, tx + 4) * mask_4d(feature_out, 2, 2, 4)
                    + shared_mem_3d(2, ty + 2, tx + 5) * mask_4d(feature_out, 2, 2, 5)
                    + shared_mem_3d(2, ty + 2, tx + 6) * mask_4d(feature_out, 2, 2, 6)
                    + shared_mem_3d(2, ty + 3, tx + 0) * mask_4d(feature_out, 2, 3, 0)
                    + shared_mem_3d(2, ty + 3, tx + 1) * mask_4d(feature_out, 2, 3, 1)
                    + shared_mem_3d(2, ty + 3, tx + 2) * mask_4d(feature_out, 2, 3, 2)
                    + shared_mem_3d(2, ty + 3, tx + 3) * mask_4d(feature_out, 2, 3, 3)
                    + shared_mem_3d(2, ty + 3, tx + 4) * mask_4d(feature_out, 2, 3, 4)
                    + shared_mem_3d(2, ty + 3, tx + 5) * mask_4d(feature_out, 2, 3, 5)
                    + shared_mem_3d(2, ty + 3, tx + 6) * mask_4d(feature_out, 2, 3, 6);
            result += shared_mem_3d(2, ty + 4, tx + 0) * mask_4d(feature_out, 2, 4, 0)
                    + shared_mem_3d(2, ty + 4, tx + 1) * mask_4d(feature_out, 2, 4, 1)
                    + shared_mem_3d(2, ty + 4, tx + 2) * mask_4d(feature_out, 2, 4, 2)
                    + shared_mem_3d(2, ty + 4, tx + 3) * mask_4d(feature_out, 2, 4, 3)
                    + shared_mem_3d(2, ty + 4, tx + 4) * mask_4d(feature_out, 2, 4, 4)
                    + shared_mem_3d(2, ty + 4, tx + 5) * mask_4d(feature_out, 2, 4, 5)
                    + shared_mem_3d(2, ty + 4, tx + 6) * mask_4d(feature_out, 2, 4, 6)
                    + shared_mem_3d(2, ty + 5, tx + 0) * mask_4d(feature_out, 2, 5, 0)
                    + shared_mem_3d(2, ty + 5, tx + 1) * mask_4d(feature_out, 2, 5, 1)
                    + shared_mem_3d(2, ty + 5, tx + 2) * mask_4d(feature_out, 2, 5, 2)
                    + shared_mem_3d(2, ty + 5, tx + 3) * mask_4d(feature_out, 2, 5, 3)
                    + shared_mem_3d(2, ty + 5, tx + 4) * mask_4d(feature_out, 2, 5, 4)
                    + shared_mem_3d(2, ty + 5, tx + 5) * mask_4d(feature_out, 2, 5, 5)
                    + shared_mem_3d(2, ty + 5, tx + 6) * mask_4d(feature_out, 2, 5, 6);
            result += shared_mem_3d(2, ty + 6, tx + 0) * mask_4d(feature_out, 2, 6, 0)
                    + shared_mem_3d(2, ty + 6, tx + 1) * mask_4d(feature_out, 2, 6, 1)
                    + shared_mem_3d(2, ty + 6, tx + 2) * mask_4d(feature_out, 2, 6, 2)
                    + shared_mem_3d(2, ty + 6, tx + 3) * mask_4d(feature_out, 2, 6, 3)
                    + shared_mem_3d(2, ty + 6, tx + 4) * mask_4d(feature_out, 2, 6, 4)
                    + shared_mem_3d(2, ty + 6, tx + 5) * mask_4d(feature_out, 2, 6, 5)
                    + shared_mem_3d(2, ty + 6, tx + 6) * mask_4d(feature_out, 2, 6, 6);
            result += shared_mem_3d(3, ty + 0, tx + 0) * mask_4d(feature_out, 3, 0, 0)
                    + shared_mem_3d(3, ty + 0, tx + 1) * mask_4d(feature_out, 3, 0, 1)
                    + shared_mem_3d(3, ty + 0, tx + 2) * mask_4d(feature_out, 3, 0, 2)
                    + shared_mem_3d(3, ty + 0, tx + 3) * mask_4d(feature_out, 3, 0, 3)
                    + shared_mem_3d(3, ty + 0, tx + 4) * mask_4d(feature_out, 3, 0, 4)
                    + shared_mem_3d(3, ty + 0, tx + 5) * mask_4d(feature_out, 3, 0, 5)
                    + shared_mem_3d(3, ty + 0, tx + 6) * mask_4d(feature_out, 3, 0, 6)
                    + shared_mem_3d(3, ty + 1, tx + 0) * mask_4d(feature_out, 3, 1, 0)
                    + shared_mem_3d(3, ty + 1, tx + 1) * mask_4d(feature_out, 3, 1, 1)
                    + shared_mem_3d(3, ty + 1, tx + 2) * mask_4d(feature_out, 3, 1, 2)
                    + shared_mem_3d(3, ty + 1, tx + 3) * mask_4d(feature_out, 3, 1, 3)
                    + shared_mem_3d(3, ty + 1, tx + 4) * mask_4d(feature_out, 3, 1, 4)
                    + shared_mem_3d(3, ty + 1, tx + 5) * mask_4d(feature_out, 3, 1, 5)
                    + shared_mem_3d(3, ty + 1, tx + 6) * mask_4d(feature_out, 3, 1, 6);
            result += shared_mem_3d(3, ty + 2, tx + 0) * mask_4d(feature_out, 3, 2, 0)
                    + shared_mem_3d(3, ty + 2, tx + 1) * mask_4d(feature_out, 3, 2, 1)
                    + shared_mem_3d(3, ty + 2, tx + 2) * mask_4d(feature_out, 3, 2, 2)
                    + shared_mem_3d(3, ty + 2, tx + 3) * mask_4d(feature_out, 3, 2, 3)
                    + shared_mem_3d(3, ty + 2, tx + 4) * mask_4d(feature_out, 3, 2, 4)
                    + shared_mem_3d(3, ty + 2, tx + 5) * mask_4d(feature_out, 3, 2, 5)
                    + shared_mem_3d(3, ty + 2, tx + 6) * mask_4d(feature_out, 3, 2, 6)
                    + shared_mem_3d(3, ty + 3, tx + 0) * mask_4d(feature_out, 3, 3, 0)
                    + shared_mem_3d(3, ty + 3, tx + 1) * mask_4d(feature_out, 3, 3, 1)
                    + shared_mem_3d(3, ty + 3, tx + 2) * mask_4d(feature_out, 3, 3, 2)
                    + shared_mem_3d(3, ty + 3, tx + 3) * mask_4d(feature_out, 3, 3, 3)
                    + shared_mem_3d(3, ty + 3, tx + 4) * mask_4d(feature_out, 3, 3, 4)
                    + shared_mem_3d(3, ty + 3, tx + 5) * mask_4d(feature_out, 3, 3, 5)
                    + shared_mem_3d(3, ty + 3, tx + 6) * mask_4d(feature_out, 3, 3, 6);
            result += shared_mem_3d(3, ty + 4, tx + 0) * mask_4d(feature_out, 3, 4, 0)
                    + shared_mem_3d(3, ty + 4, tx + 1) * mask_4d(feature_out, 3, 4, 1)
                    + shared_mem_3d(3, ty + 4, tx + 2) * mask_4d(feature_out, 3, 4, 2)
                    + shared_mem_3d(3, ty + 4, tx + 3) * mask_4d(feature_out, 3, 4, 3)
                    + shared_mem_3d(3, ty + 4, tx + 4) * mask_4d(feature_out, 3, 4, 4)
                    + shared_mem_3d(3, ty + 4, tx + 5) * mask_4d(feature_out, 3, 4, 5)
                    + shared_mem_3d(3, ty + 4, tx + 6) * mask_4d(feature_out, 3, 4, 6)
                    + shared_mem_3d(3, ty + 5, tx + 0) * mask_4d(feature_out, 3, 5, 0)
                    + shared_mem_3d(3, ty + 5, tx + 1) * mask_4d(feature_out, 3, 5, 1)
                    + shared_mem_3d(3, ty + 5, tx + 2) * mask_4d(feature_out, 3, 5, 2)
                    + shared_mem_3d(3, ty + 5, tx + 3) * mask_4d(feature_out, 3, 5, 3)
                    + shared_mem_3d(3, ty + 5, tx + 4) * mask_4d(feature_out, 3, 5, 4)
                    + shared_mem_3d(3, ty + 5, tx + 5) * mask_4d(feature_out, 3, 5, 5)
                    + shared_mem_3d(3, ty + 5, tx + 6) * mask_4d(feature_out, 3, 5, 6);
            result += shared_mem_3d(3, ty + 6, tx + 0) * mask_4d(feature_out, 3, 6, 0)
                    + shared_mem_3d(3, ty + 6, tx + 1) * mask_4d(feature_out, 3, 6, 1)
                    + shared_mem_3d(3, ty + 6, tx + 2) * mask_4d(feature_out, 3, 6, 2)
                    + shared_mem_3d(3, ty + 6, tx + 3) * mask_4d(feature_out, 3, 6, 3)
                    + shared_mem_3d(3, ty + 6, tx + 4) * mask_4d(feature_out, 3, 6, 4)
                    + shared_mem_3d(3, ty + 6, tx + 5) * mask_4d(feature_out, 3, 6, 5)
                    + shared_mem_3d(3, ty + 6, tx + 6) * mask_4d(feature_out, 3, 6, 6);
            out_4d(batch_out, feature_out, h_out, w_out) = result;
        }
    }
    
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
    cudaMalloc((void **) device_output_ptr, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float));
    cudaMalloc((void **) device_input_ptr, (Batch * Channel * Height * Width)*sizeof(float));
    // cudaMalloc((void **) device_mask_ptr, (Map_out * Channel * K * K)*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, (Batch * Channel * Height * Width)*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, (Map_out * Channel * K * K)*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(Constant_mask, host_mask, (Map_out * Channel * K * K)*sizeof(float));
    printf("Kernel size:%d\n", (Map_out * Channel * K * K));
    printf("Input size:%d\n", (Batch * Channel * Height * Width));
    printf("kernel width:%d\n",K);
    // printf("Kernel size:", sizeof(host_mask));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // std::cout<<"kernel size"<< K <<endl;
    // Set the kernel dimensions and call the kernel
    dim3 dimGrid(Map_out, ceil((float)(Height - K + 1)/BLOCK_SIZE)*ceil((float)(Width - K + 1)/BLOCK_SIZE), Batch);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    printf("Mapout:%d\n", Map_out);
    printf("Channel:%d\n", Channel);
    // const int layer = 0;
    // if 
    // we need a shared space of C*inputwidth_of_block*inputwidth_of_block
    conv_forward_kernel<<<dimGrid, dimBlock, Channel*(BLOCK_SIZE + K - 1)*(BLOCK_SIZE + K - 1)*sizeof(float)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    cudaMemcpy(host_output, device_output, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output);
    cudaFree(device_input);
    cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}

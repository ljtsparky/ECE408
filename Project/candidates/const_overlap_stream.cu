#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define BLOCK_SIZE 16 //@@ You can change this

__constant__ float Constant_mask[6000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Constant_mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

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

    if (h_out < Height_out && w_out < Width_out){
        float result = 0;
        for (int c = 0; c < Channel; c++){
            for (int ky = 0; ky < K; ky++){
                for (int kx = 0; kx < K; kx++){
                    result += in_4d(batch_out, c, h_out + ky, w_out + kx) * mask_4d(feature_out, c, ky, kx);
                }
            }
        }
        out_4d(batch_out, feature_out, h_out, w_out) = result;
    }
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
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
    // // cudaMalloc((void **) device_mask_ptr, (Map_out * Channel * K * K)*sizeof(float));
    // cudaMemcpy(*device_input_ptr_ptr, host_input, (Batch * Channel * Height * Width)*sizeof(float), cudaMemcpyHostToDevice);
    // // cudaMemcpy(*device_mask_ptr, host_mask, (Map_out * Channel * K * K)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Constant_mask, host_mask, (Map_out * Channel * K * K)*sizeof(float));
    cudaStream_t stream0, stream1, stream2, stream3, stream4, stream5, stream6, stream7, stream8, stream9; //stream10, stream11; //stream12, stream13, stream14, stream15;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);
    cudaStreamCreate(&stream7);
    cudaStreamCreate(&stream8);
    cudaStreamCreate(&stream9);
    // cudaStreamCreate(&stream10);
    // cudaStreamCreate(&stream11);
    // cudaStreamCreate(&stream12);
    // cudaStreamCreate(&stream13);
    // cudaStreamCreate(&stream14);
    // cudaStreamCreate(&stream15);
    int SegNum = 10;
    int Input_segment_size = Channel*Height*Width;
    int Output_segment_size = Map_out*(Height-K+1)*(Width-K+1);
    dim3 dimGrid(Map_out, ceil((float)(Height - K + 1)/BLOCK_SIZE)*ceil((float)(Width - K + 1)/BLOCK_SIZE), SegNum);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output_ptr, device_input_ptr, device_mask, Batch, Map_out, Channel, Height, Width, K);
    for (int i = 0; i < Batch; i += SegNum*10){
        cudaMemcpyAsync(*device_input_ptr + (i + 0 * SegNum) *Input_segment_size, host_input + (i + 0 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(*device_input_ptr + (i + 1 * SegNum) *Input_segment_size, host_input + (i + 1 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(*device_input_ptr + (i + 2 * SegNum) *Input_segment_size, host_input + (i + 2 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(*device_input_ptr + (i + 3 * SegNum) *Input_segment_size, host_input + (i + 3 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream3);
        cudaMemcpyAsync(*device_input_ptr + (i + 4 * SegNum) *Input_segment_size, host_input + (i + 4 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream4);
        cudaMemcpyAsync(*device_input_ptr + (i + 5 * SegNum) *Input_segment_size, host_input + (i + 5 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream5);
        cudaMemcpyAsync(*device_input_ptr + (i + 6 * SegNum) *Input_segment_size, host_input + (i + 6 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream6);
        cudaMemcpyAsync(*device_input_ptr + (i + 7 * SegNum) *Input_segment_size, host_input + (i + 7 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream7);
        cudaMemcpyAsync(*device_input_ptr + (i + 8 * SegNum) *Input_segment_size, host_input + (i + 8 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream8);
        cudaMemcpyAsync(*device_input_ptr + (i + 9 * SegNum) *Input_segment_size, host_input + (i + 9 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream9);
        // cudaMemcpyAsync(*device_input_ptr + (i + 10 * SegNum) *Input_segment_size, host_input + (i + 10 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream10);
        // cudaMemcpyAsync(*device_input_ptr + (i + 11 * SegNum) *Input_segment_size, host_input + (i + 11 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream11);
        // cudaMemcpyAsync(*device_input_ptr + (i + 12 * SegNum) *Input_segment_size, host_input + (i + 12 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream12);
        // cudaMemcpyAsync(*device_input_ptr + (i + 13 * SegNum) *Input_segment_size, host_input + (i + 13 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream13);
        // cudaMemcpyAsync(*device_input_ptr + (i + 14 * SegNum) *Input_segment_size, host_input + (i + 14 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream14);
        // cudaMemcpyAsync(*device_input_ptr + (i + 15 * SegNum) *Input_segment_size, host_input + (i + 15 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream15);
        
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream0>>>(*device_output_ptr + (i + 0 * SegNum) * Output_segment_size, *device_input_ptr + (i + 0 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream1>>>(*device_output_ptr + (i + 1 * SegNum) * Output_segment_size, *device_input_ptr + (i + 1 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream2>>>(*device_output_ptr + (i + 2 * SegNum) * Output_segment_size, *device_input_ptr + (i + 2 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream3>>>(*device_output_ptr + (i + 3 * SegNum) * Output_segment_size, *device_input_ptr + (i + 3 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream4>>>(*device_output_ptr + (i + 4 * SegNum) * Output_segment_size, *device_input_ptr + (i + 4 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream5>>>(*device_output_ptr + (i + 5 * SegNum) * Output_segment_size, *device_input_ptr + (i + 5 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream6>>>(*device_output_ptr + (i + 6 * SegNum) * Output_segment_size, *device_input_ptr + (i + 6 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream7>>>(*device_output_ptr + (i + 7 * SegNum) * Output_segment_size, *device_input_ptr + (i + 7 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream8>>>(*device_output_ptr + (i + 8 * SegNum) * Output_segment_size, *device_input_ptr + (i + 8 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        conv_forward_kernel<<<dimGrid, dimBlock, 0, stream9>>>(*device_output_ptr + (i + 9 * SegNum) * Output_segment_size, *device_input_ptr + (i + 9 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        // conv_forward_kernel<<<dimGrid, dimBlock, 0, stream10>>>(*device_output_ptr + (i + 10 * SegNum) * Output_segment_size, *device_input_ptr + (i + 10 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        // conv_forward_kernel<<<dimGrid, dimBlock, 0, stream11>>>(*device_output_ptr + (i + 11 * SegNum) * Output_segment_size, *device_input_ptr + (i + 11 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        // conv_forward_kernel<<<dimGrid, dimBlock, 0, stream4>>>(*device_output_ptr + (i + 12 * SegNum) * Output_segment_size, *device_input_ptr + (i + 12 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        // conv_forward_kernel<<<dimGrid, dimBlock, 0, stream5>>>(*device_output_ptr + (i + 13 * SegNum) * Output_segment_size, *device_input_ptr + (i + 13 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        // conv_forward_kernel<<<dimGrid, dimBlock, 0, stream6>>>(*device_output_ptr + (i + 14 * SegNum) * Output_segment_size, *device_input_ptr + (i + 14 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
        // conv_forward_kernel<<<dimGrid, dimBlock, 0, stream7>>>(*device_output_ptr + (i + 15 * SegNum) * Output_segment_size, *device_input_ptr + (i + 15 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);

        cudaMemcpyAsync(host_output + (i + 0 * SegNum) * Output_segment_size, *device_output_ptr + (i + 0 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(host_output + (i + 1 * SegNum) * Output_segment_size, *device_output_ptr + (i + 1 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(host_output + (i + 2 * SegNum) * Output_segment_size, *device_output_ptr + (i + 2 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
        cudaMemcpyAsync(host_output + (i + 3 * SegNum) * Output_segment_size, *device_output_ptr + (i + 3 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream3);
        cudaMemcpyAsync(host_output + (i + 4 * SegNum) * Output_segment_size, *device_output_ptr + (i + 4 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream4);
        cudaMemcpyAsync(host_output + (i + 5 * SegNum) * Output_segment_size, *device_output_ptr + (i + 5 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream5);
        cudaMemcpyAsync(host_output + (i + 6 * SegNum) * Output_segment_size, *device_output_ptr + (i + 6 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream6);
        cudaMemcpyAsync(host_output + (i + 7 * SegNum) * Output_segment_size, *device_output_ptr + (i + 7 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream7);
        cudaMemcpyAsync(host_output + (i + 8 * SegNum) * Output_segment_size, *device_output_ptr + (i + 8 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream8);
        cudaMemcpyAsync(host_output + (i + 9 * SegNum) * Output_segment_size, *device_output_ptr + (i + 9 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream9);
        // cudaMemcpyAsync(host_output + (i + 10 * SegNum) * Output_segment_size, *device_output_ptr + (i + 10 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream10);
        // cudaMemcpyAsync(host_output + (i + 11 * SegNum) * Output_segment_size, *device_output_ptr + (i + 11 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream11);
        // cudaMemcpyAsync(host_output + (i + 12 * SegNum) * Output_segment_size, *device_output_ptr + (i + 12 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream12);
        // cudaMemcpyAsync(host_output + (i + 13 * SegNum) * Output_segment_size, *device_output_ptr + (i + 13 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream5);
        // cudaMemcpyAsync(host_output + (i + 14 * SegNum) * Output_segment_size, *device_output_ptr + (i + 14 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream6);
        // cudaMemcpyAsync(host_output + (i + 15 * SegNum) * Output_segment_size, *device_output_ptr + (i + 15 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream7);
    }
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output_ptr, const float *device_input_ptr, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{

    // std::cout<<"kernel size"<< K <<endl;
    // Set the kernel dimensions and call the kernel
    // int Input_segment_size = Channel*Height*Width;
    // int Output_segment_size = Map_out*(Height-K+1)*(Width-K+1)
    // dim3 dimGrid(Map_out, ceil((float)(Height - K + 1)/BLOCK_SIZE)*ceil((float)(Width - K + 1)/BLOCK_SIZE), SegNum);
    // dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // // conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output_ptr, device_input_ptr, device_mask, Batch, Map_out, Channel, Height, Width, K);
    // for (int i = 0; i < Batch; i += SegNum*8){
    //     cudaMemcpyAsync(*device_input_ptr + (i + 0 * SegNum) *Input_segment_size, host_input + (i + 0 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream0);
    //     cudaMemcpyAsync(*device_input_ptr + (i + 1 * SegNum) *Input_segment_size, host_input + (i + 1 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream1);
    //     cudaMemcpyAsync(*device_input_ptr + (i + 2 * SegNum) *Input_segment_size, host_input + (i + 2 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream2);
    //     cudaMemcpyAsync(*device_input_ptr + (i + 3 * SegNum) *Input_segment_size, host_input + (i + 3 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream3);
    //     cudaMemcpyAsync(*device_input_ptr + (i + 4 * SegNum) *Input_segment_size, host_input + (i + 4 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream4);
    //     cudaMemcpyAsync(*device_input_ptr + (i + 5 * SegNum) *Input_segment_size, host_input + (i + 5 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream5);
    //     cudaMemcpyAsync(*device_input_ptr + (i + 6 * SegNum) *Input_segment_size, host_input + (i + 6 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream6);
    //     cudaMemcpyAsync(*device_input_ptr + (i + 7 * SegNum) *Input_segment_size, host_input + (i + 7 * SegNum) *Input_segment_size, SegNum *Input_segment_size * sizeof(float), cudaMemcpyHostToDevice, stream7);

    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream0>>>(*device_output_ptr + (i + 0 * SegNum) * Output_segment_size, *device_input_ptr + (i + 0 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream1>>>(*device_output_ptr + (i + 1 * SegNum) * Output_segment_size, *device_input_ptr + (i + 1 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream2>>>(*device_output_ptr + (i + 2 * SegNum) * Output_segment_size, *device_input_ptr + (i + 2 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream3>>>(*device_output_ptr + (i + 3 * SegNum) * Output_segment_size, *device_input_ptr + (i + 3 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream4>>>(*device_output_ptr + (i + 4 * SegNum) * Output_segment_size, *device_input_ptr + (i + 4 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream5>>>(*device_output_ptr + (i + 5 * SegNum) * Output_segment_size, *device_input_ptr + (i + 5 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream6>>>(*device_output_ptr + (i + 6 * SegNum) * Output_segment_size, *device_input_ptr + (i + 6 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);
    //     conv_forward_kernel<<<dimGrid, dimBlock, 0, stream7>>>(*device_output_ptr + (i + 7 * SegNum) * Output_segment_size, *device_input_ptr + (i + 7 * SegNum) *Input_segment_size, *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);

    //     cudaMemcpyAsync(host_input + (i + 0 * SegNum) * Output_segment_size, *device_output_ptr + (i + 0 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream0);
    //     cudaMemcpyAsync(host_input + (i + 1 * SegNum) * Output_segment_size, *device_output_ptr + (i + 1 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    //     cudaMemcpyAsync(host_input + (i + 2 * SegNum) * Output_segment_size, *device_output_ptr + (i + 2 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    //     cudaMemcpyAsync(host_input + (i + 3 * SegNum) * Output_segment_size, *device_output_ptr + (i + 3 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream3);
    //     cudaMemcpyAsync(host_input + (i + 4 * SegNum) * Output_segment_size, *device_output_ptr + (i + 4 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream4);
    //     cudaMemcpyAsync(host_input + (i + 5 * SegNum) * Output_segment_size, *device_output_ptr + (i + 5 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream5);
    //     cudaMemcpyAsync(host_input + (i + 6 * SegNum) * Output_segment_size, *device_output_ptr + (i + 6 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream6);
    //     cudaMemcpyAsync(host_input + (i + 7 * SegNum) * Output_segment_size, *device_output_ptr + (i + 7 * SegNum) * Output_segment_size, SegNum * Output_segment_size * sizeof(float), cudaMemcpyDeviceToHost, stream7);
    // }
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output_ptr, float *device_input_ptr, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    // cudaMemcpy(host_output, device_output_ptr, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(device_output_ptr);
    cudaFree(device_input_ptr);
    // cudaFree(device_mask);
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

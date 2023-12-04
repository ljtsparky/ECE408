#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include "cuda_fp16.h"
#define BLOCK_SIZE1 20 //@@ You can change this
#define BLOCK_SIZE2 16
// __constant__ float Constant_mask[3136];
__constant__ __half2 Constant_mask_rightzero[1792]; //64*7*4
__constant__ __half2 Constant_mask_leftzero[1792]; //64*7*4
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

    int BLOCK_SIZE = 16;
    if (Map_out==4){
        BLOCK_SIZE = BLOCK_SIZE1;
    }
    else{
        BLOCK_SIZE = BLOCK_SIZE2;
    }
    const int shared_mem_size_x = (BLOCK_SIZE + K -1) / 2;
    const int shared_mem_size_y = BLOCK_SIZE + K - 1;
    extern __shared__ __half2 shared_mem[];
    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d_lz(i3, i2, i1, i0) Constant_mask_leftzero[(i3) * (Channel * 28) + (i2) * 28 + (i1) * (4) + i0]
    #define mask_4d_rz(i3, i2, i1, i0) Constant_mask_rightzero[(i3) * (Channel * 28) + (i2) * 28 + (i1) * (4) + i0]
    // #define mask_4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define shared_mem_3d(i2,i1,i0) shared_mem[ (i2) * (shared_mem_size_y*shared_mem_size_x) + (i1)*shared_mem_size_x + i0]
    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int blocknum_per_row = (Width_out - 1)/BLOCK_SIZE + 1; 
    int w_out = BLOCK_SIZE * (by % blocknum_per_row) + tx; // width out for this thread would depend on remain 
    int h_out = BLOCK_SIZE * (by / blocknum_per_row) + ty; // height for this thread would depend on how many rows we've come through
    int batch_out = bz; //bz is for batches
    int feature_out = bx; //bx is for different output features

    int block_lu_x = BLOCK_SIZE * (by % blocknum_per_row);
    int block_lu_y = BLOCK_SIZE * (by / blocknum_per_row);
    // one channel, K=7
    if (Map_out == 4){
        if (tx%2==0){
            for ( int i = ty; i < shared_mem_size_y; i+=BLOCK_SIZE){
                for (int j = tx;j < shared_mem_size_y; j+=BLOCK_SIZE){
                    if (block_lu_x + tx < Width && block_lu_y + ty < Height){
                        float a = in_4d(batch_out, 0, block_lu_y + i, block_lu_x + j);
                        float b = in_4d(batch_out, 0, block_lu_y + i, block_lu_x + j + 1);
                        shared_mem_3d(0, i, j/2) = __floats2half2_rn(b, a);
                    }
                }
            }
        }
        __syncthreads();

        if (h_out < Height_out && w_out < Width_out){
            if (tx%2==0){
                __half2 temp = __floats2half2_rn(0.0f, 0.0f);
                for (int c=0;c<Channel;c++){
                    for (int ky=0;ky<K;ky++){
                        for (int kx=0;kx<4;kx++){
                            // __half2 mul_result= __hmul2(mask_4d_rz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx));
                            // temp = __hadd2(mul_result, temp);
                            temp = __hfma2(mask_4d_rz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx), temp);
                        }
                    }
                }
                __half a = __low2half(temp);
                __half b = __high2half(temp);
                __half c = __hadd(a, b);
                out_4d(batch_out, feature_out, h_out, w_out) = __half2float(c);
            }
            else{
                __half2 temp = __floats2half2_rn(0.0f, 0.0f);
                for (int c=0;c<Channel;c++){
                    for (int ky=0;ky<K;ky++){
                        for (int kx=0;kx<4;kx++){
                            // __half2 mul_result= __hmul2(mask_4d_lz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx));
                            // temp = __hadd2(mul_result, temp);
                            temp = __hfma2(mask_4d_lz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx), temp);
                        }
                    }
                }
                __half a = __low2half(temp);
                __half b = __high2half(temp);
                __half c = __hadd(a, b);
                out_4d(batch_out, feature_out, h_out, w_out) = __half2float(c);
            }
        }
    }
    else{
        if (tx%2==0){
            for (int c=0;c<Channel;c++){
                for ( int i = ty; i < shared_mem_size_y; i+=BLOCK_SIZE){
                    for (int j = tx;j < shared_mem_size_y; j+=BLOCK_SIZE){
                        if (block_lu_x + tx < Width && block_lu_y + ty < Height){
                            float a = in_4d(batch_out, c, block_lu_y + i, block_lu_x + j);
                            float b = in_4d(batch_out, c, block_lu_y + i, block_lu_x + j + 1);
                            shared_mem_3d(c, i, j/2) = __floats2half2_rn(b, a);
                        }
                    }
                }
            }
            
        }
        __syncthreads();

        if (h_out < Height_out && w_out < Width_out){
            if (tx%2==0){
                __half2 temp = __floats2half2_rn(0.0f, 0.0f);
                for (int c=0;c<Channel;c++){
                    for (int ky=0;ky<K;ky++){
                        for (int kx=0;kx<4;kx++){
                            // __half2 mul_result= __hmul2(mask_4d_rz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx));
                            // temp = __hadd2(mul_result, temp);
                            temp = __hfma2(mask_4d_rz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx), temp);
                        }
                    }
                }
                __half a = __low2half(temp);
                __half b = __high2half(temp);
                __half c = __hadd(a, b);
                out_4d(batch_out, feature_out, h_out, w_out) = __half2float(c);
            }
            else{
                __half2 temp = __floats2half2_rn(0.0f, 0.0f);
                for (int c=0;c<Channel;c++){
                    for (int ky=0;ky<K;ky++){
                        for (int kx=0;kx<4;kx++){
                            // __half2 mul_result= __hmul2(mask_4d_lz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx));
                            // temp = __hadd2(mul_result, temp);
                            temp = __hfma2(mask_4d_lz(feature_out,c,ky,kx), shared_mem_3d(c, ty+ky, tx/2+kx), temp);
                        }
                    }
                }
                __half a = __low2half(temp);
                __half b = __high2half(temp);
                __half c = __hadd(a, b);
                out_4d(batch_out, feature_out, h_out, w_out) = __half2float(c);
            }
        }
    }
    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // __half2 input_half2[Batch * Channel * Height * Width / 2];
    // for (int i=0;i < Batch * Channel * Height * Width;i+=2){
    //     float a = host_input[i];
    //     float b = host_input[i+1];
    //     input_half2[i/2]=__floats2half2_rn(a,b);
    // }
    __half2 host_mask_half2_rightzero[Map_out * Channel * 28];
    __half2 host_mask_half2_leftzero[Map_out * Channel * 28];
    for (int i=0;i<Map_out;i++){
        for (int j=0;j<Channel;j++){
            for (int k=0;k<49;k+=1){
                if (k%7==6 || k%7==2 || k%7==4 || k%7==0){
                    float a = host_mask[i*Channel*49+j*49+k];
                    float b = 0;
                    if (k%7==0 || k%7==2 || k%7==4){
                        b =host_mask[i*Channel*49+j*49+k+1];
                    }
                    host_mask_half2_rightzero[i*Channel*28 + j*28 + (k / 7) * 4 + (k%7)/2 ]=__floats2half2_rn(b,a);
                    if (k%7==0){
                        b = 0;
                    }
                    if (k%7 == 2 || k%7 == 4 || k%7 ==6 ){
                        b = host_mask[i*Channel*49+j*49+k-1];
                    }
                    host_mask_half2_leftzero[i*Channel*28 + j*28 + (k / 7) * 4 + (k%7)/2]=__floats2half2_rn(a,b);
                }
            }
        }
    }
    // since input is 8686 or 40 40, so we can directly divide by 2
    printf("K:%d\n", K);
    printf("Mapout:%d\n", Map_out);
    printf("Height:%d\n", Height);
    printf("Width:%d\n", Width);
    printf("test:%d\n", 1/8);
    // printf("Input size:%d\n", (Batch * Channel * Height * Width));
    // printf("kernel width:%d\n",K);
    // printf("Kernel size:", sizeof(host_mask));
    cudaMalloc((void **) device_input_ptr, (Batch * Channel * Height * Width )*sizeof(float));
    cudaMalloc((void **) device_output_ptr, (Batch * Map_out * (Height - K + 1) * (Width - K + 1))*sizeof(float));

    cudaMemcpy(*device_input_ptr, host_input, (Batch * Channel * Height * Width)*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Constant_mask_rightzero, host_mask_half2_rightzero, (Map_out * Channel * 28 )*sizeof(__half2));
    cudaMemcpyToSymbol(Constant_mask_leftzero, host_mask_half2_leftzero, (Map_out * Channel * 28 )*sizeof(__half2));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // std::cout<<"kernel size"<< K <<endl;
    // Set the kernel dimensions and call the kernel
    int BLOCK_SIZE=16;
    if (Height==86){
        BLOCK_SIZE=20;
    }
    else{
        BLOCK_SIZE=16;
    }
    dim3 dimGrid(Map_out, ceil((float)(Height - K + 1)/BLOCK_SIZE)*ceil((float)(Width - K + 1)/BLOCK_SIZE), Batch);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    // we need a shared space of C*inputwidth_of_block*inputwidth_of_block
    conv_forward_kernel<<<dimGrid, dimBlock, Channel*(BLOCK_SIZE + K - 1)*(BLOCK_SIZE + K - 1)/2*sizeof(__half2)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
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

// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void finish(float *output, int len, float *auxarray) {
  int tid=threadIdx.x;
  int bid=blockIdx.x;
  int start=2*bid*BLOCK_SIZE;
  int tobeadded = bid-1;
  __syncthreads();
  while (tobeadded>=0){
    if (start+tid<len){
      output[start+tid]+=auxarray[tobeadded];
    }
    __syncthreads();
    tobeadded-=1;
  }
  __syncthreads();
}

__global__ void scan(float *input, float *output, int len, float *auxarray) {
  __shared__ float T[2*BLOCK_SIZE];
  int t = threadIdx.x;
  int bx = blockIdx.x;
  int START = bx*BLOCK_SIZE*2;
  for (int i = START; i<START + 2*BLOCK_SIZE ; i++){
    if (i<len){
      T[i-START]=input[i];
    }
    else{
      T[i-START]=0;
    }
  }
  int stride=1;
  while(stride<2*BLOCK_SIZE){
    __syncthreads();
    int index= (t + 1)*stride*2-1;
    if (index<2*BLOCK_SIZE && (index-stride)>=0){
      T[index]+=T[index-stride];
    }
    stride=stride*2;
  }
  __syncthreads();
  stride = BLOCK_SIZE/2;
  while(stride>0){
    __syncthreads();
    int index = (t+1)*stride*2-1;
    if ((index+stride)<2*BLOCK_SIZE){
      T[index+stride]+=T[index];
    }
    stride=stride/2;
  }
  __syncthreads();
  for (int j=0; j<2*BLOCK_SIZE;j++){
    if (START+j<len){
      output[START+j]=T[j];
    }
  }
  auxarray[bx]=T[2*BLOCK_SIZE-1];
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *auxarray;
  int numElements; // number of elements in the list
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  cudaMalloc((void **)&auxarray, ceil(numElements/(2*BLOCK_SIZE*1.0)) * sizeof(float));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid( ceil(numElements/(2*BLOCK_SIZE*1.0) ), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);
  dim3 dimBlock2(2*BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements, auxarray);
  cudaDeviceSynchronize();
  finish<<<dimGrid, dimBlock2>>>(deviceOutput, numElements, auxarray);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

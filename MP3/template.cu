
#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

#define block_width 32

// Compute C = A * B
__global__ 
void MatrixMulKernel ( float *A, float *B, float *C,
                                int numARows, int numAColumns, 
                                int numBRows, int numBColumns, 
                                int numCRows, int numCColumns){
  __shared__ float subTileM[block_width][block_width];
  __shared__ float subTileN[block_width][block_width];
  int bx = blockIdx.x; 
  int by = blockIdx.y;
  int tx = threadIdx.x; 
  int ty = threadIdx.y;
  // Identify the row and column of the P element to work on
  int Row = by * block_width + ty;
  int Col = bx * block_width + tx;
  float Pvalue = 0;
  // Loop over the M and N tiles required to compute the P element
  // The code assumes that the Width is a multiple of TILE_WIDTH!
  for (int q = 0; q < (numAColumns-1)/block_width+1; ++q) { // q is the column of block
  // Collaborative loading of M and N tiles into shared memory
    if ( (q*block_width+tx) < numAColumns && Row<numARows){
      subTileM[ty][tx] = A[Row*numAColumns + q*block_width+tx];
    } else {
      subTileM[ty][tx] = 0;
    }
    if (q*block_width+ty < numBRows && Col < numBColumns ){
      subTileN[ty][tx] = B[(q*block_width+ty)*numBColumns+Col];
    } else {
      subTileN[ty][tx] = 0;
    }
    __syncthreads();
    if (Row<numCRows && Col<numCColumns){
      for (int k = 0; k < block_width; ++k){
        Pvalue += subTileM[ty][k] * subTileN[k][tx];
      }
    }
    __syncthreads();
  }
  if (Row<numCRows && Col<numCColumns){
    C[Row*numCColumns+Col] = Pvalue;
  }
};

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  // int dev_count;
  // cudaGetDeviceCount( &dev_count);
  // cudaDeviceProp dev_prop;
  // for (i = 0; i < dev_count; i++) {
  //   cudaGetDeviceProperties( &dev_prop, i);
  //   // decide if device has sufficient resources and capabilities
  // }
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  int size_to_allocate_A = sizeof(float) * numARows*numAColumns;
  int size_to_allocate_B = sizeof(float) * numBRows*numBColumns;
  int size_to_allocate_C = sizeof(float)*numCRows*numCColumns;
  //@@ Allocate the hostC matrix
  hostC=(float *)malloc(size_to_allocate_C);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  cudaMalloc((void **) &deviceA, size_to_allocate_A);
  cudaMalloc((void **) &deviceB, size_to_allocate_B);
  cudaMalloc((void **) &deviceC, size_to_allocate_C);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  cudaMemcpy(deviceA, hostA, size_to_allocate_A, cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, size_to_allocate_B, cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil((1.0*numCColumns)/block_width),ceil((1.0*numCRows)/block_width), 1);
  dim3 dimBlock(block_width, block_width, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  matrixMultiplyShared<<<dimGrid,dimBlock>>>(deviceA,deviceB,deviceC,numARows,numAColumns,numBRows,numBColumns,numCRows,numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC, deviceC, size_to_allocate_C, cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}

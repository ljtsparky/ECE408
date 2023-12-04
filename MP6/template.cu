// Histogram Equalization

#include <wb.h>

<<<<<<< Updated upstream
#define HISTOGRAM_LENGTH 256

//@@ insert code here
=======
#define BLOCK_SIZE 32
#define HISTOGRAM_LENGTH 256

//@@ insert code here
__global__ void float2unsignedchar(float *inputImage, unsigned char *ucharImage, int height, int width ){

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;
  int channel = blockIdx.z;
  if (tx < width && ty < height){
    int channel_position = channel*height*width + ty*width +tx;
    ucharImage[channel_position] = (unsigned char) (255*inputImage[channel_position]);
  }
}

__global__ void RGB2Grayscale(unsigned char *ucharImagergb, unsigned char *ucharImagegray, int height, int width){

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;
  if (tx < width && ty < height){
    int idx = ty*width + tx;
    unsigned char r = ucharImagergb[3*idx];
    unsigned char g = ucharImagergb[3*idx + 1];
    unsigned char b = ucharImagergb[3*idx + 2];
    ucharImagegray[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void compute_histogram_of_grayimage (unsigned char *grayimage, unsigned int *histo, int height, int width){

  __shared__ unsigned int histo_private[HISTOGRAM_LENGTH];
  int i = threadIdx.x + threadIdx.y*blockDim.x;

  if (i < 256){
    histo_private[i] = 0;
  }

  __syncthreads();

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;

  if (tx < width && ty < height){
    int idx = ty*width + tx;
    atomicAdd( &(histo_private[grayimage[idx]]), 1);
  }

  __syncthreads();

  if (i < 256){
    atomicAdd( &(histo[i]),histo_private[i]);
  }
}

__global__ void compute_cdf(unsigned int *histo, float *cdfout, int height, int width){

  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int i = threadIdx.x + blockDim.x*blockIdx.x;
  if (i < HISTOGRAM_LENGTH){
    cdf[i] = histo[i];
  }

  unsigned int stride = 1;
  while (stride <= HISTOGRAM_LENGTH/2){
    __syncthreads();
    int idx = (threadIdx.x+1)*stride*2 - 1;
    if(idx < HISTOGRAM_LENGTH ){
      cdf[idx] += cdf[idx - stride];
    }
    stride = stride * 2;
  }
  __syncthreads();
  stride = HISTOGRAM_LENGTH/4;
  while (stride > 0){
    __syncthreads();
    int idx = (threadIdx.x+1)*stride*2 - 1;
    if((idx+stride) < HISTOGRAM_LENGTH){
      cdf[idx+stride] += cdf[idx];
    }
    stride = stride / 2;
  }
  __syncthreads();
  if (i < HISTOGRAM_LENGTH){
    cdfout[i] = cdf[i]/((float)(width*height));
  }
}


__global__ void equalization(unsigned char *image, float *cdf, float *final_image, int height, int width){

  int tx = threadIdx.x + blockDim.x*blockIdx.x;
  int ty = threadIdx.y + blockDim.y*blockIdx.y;
  int channel=blockIdx.z;
  float cdfmin=cdf[0];
  if (tx < width && ty < height){
    int idx = channel * (width * height) + ty * (width) + tx;
    float x = 255*(cdf[image[idx]] - cdfmin)/(1.0 - cdfmin);
    float clamp = min(max(x, 0.0), 255.0);
    image[idx] = (unsigned char)(clamp);
  }
  __syncthreads();
  if (tx < width && ty < height){
    int idx = channel*height*width + ty*width + tx;
    final_image[idx] = (float) (image[idx]/255.0);
  }
}



>>>>>>> Stashed changes

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
<<<<<<< Updated upstream
=======
  float *deviceInput; float *deviceOutput;
  unsigned char *deviceUnsignedcharRGBImage;
  unsigned char *deviceUnsignedcharGRAYImage;
  unsigned int *deviceHisto;  float *deviceCdf;
>>>>>>> Stashed changes

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
<<<<<<< Updated upstream

  wbSolution(args, outputImage);

  //@@ insert code here

=======
  cudaMalloc((void**) &deviceInput, imageChannels*imageWidth*imageHeight * sizeof(float));
  cudaMalloc((void**) &deviceOutput, imageChannels*imageWidth*imageHeight * sizeof(float));
  cudaMalloc((void**) &deviceUnsignedcharRGBImage, imageChannels*imageWidth*imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &deviceUnsignedcharGRAYImage, imageWidth * imageHeight * sizeof(unsigned char));
  cudaMalloc((void**) &deviceHisto, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMalloc((void**) &deviceCdf, HISTOGRAM_LENGTH * sizeof(float));
  //set to zero initially
  cudaMemset((void *) deviceHisto, 0, HISTOGRAM_LENGTH * sizeof(unsigned int));
  cudaMemset((void *) deviceCdf, 0, HISTOGRAM_LENGTH * sizeof(float));
  cudaMemcpy(deviceInput, hostInputImageData, imageChannels*imageWidth*imageHeight * sizeof(float), cudaMemcpyHostToDevice);

  //@@ insert code here
  // turn to unsignedchar
  dim3 DimGrid = dim3(ceil(imageWidth/(1.0*BLOCK_SIZE)),ceil(imageHeight/(1.0*BLOCK_SIZE)),imageChannels);
  dim3 DimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  float2unsignedchar<<<DimGrid, DimBlock>>> (deviceInput,deviceUnsignedcharRGBImage,imageHeight, imageWidth);
  cudaDeviceSynchronize();

  //turn RGB to GRAY
  DimGrid = dim3(ceil(imageWidth/(1.0*BLOCK_SIZE)),ceil(imageHeight/(1.0*BLOCK_SIZE)),1);
  DimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  RGB2Grayscale<<<DimGrid,DimBlock>>>(deviceUnsignedcharRGBImage,deviceUnsignedcharGRAYImage,imageHeight, imageWidth);
  cudaDeviceSynchronize();
  // turn gray to histogram
  compute_histogram_of_grayimage<<<DimGrid,DimBlock>>>(deviceUnsignedcharGRAYImage,deviceHisto,imageHeight, imageWidth);
  cudaDeviceSynchronize();

  DimGrid  = dim3(1, 1, 1);
  DimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  compute_cdf<<<DimGrid,DimBlock>>>(deviceHisto,deviceCdf,imageHeight, imageWidth);
  cudaDeviceSynchronize();

  DimGrid = dim3(ceil(imageWidth/(1.0*BLOCK_SIZE)),ceil(imageHeight/(1.0*BLOCK_SIZE)),imageChannels);
  DimBlock = dim3(BLOCK_SIZE,BLOCK_SIZE,1);
  equalization<<<DimGrid,DimBlock>>>(deviceUnsignedcharRGBImage, deviceCdf, deviceOutput, imageHeight, imageWidth);
  cudaDeviceSynchronize();
  //final copy to host output
  cudaMemcpy(hostOutputImageData, deviceOutput, imageChannels*imageWidth*imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  cudaFree(deviceInput); cudaFree(deviceOutput); cudaFree(deviceHisto); cudaFree(deviceCdf); 
  cudaFree(deviceUnsignedcharGRAYImage); cudaFree(deviceUnsignedcharRGBImage);
  free(hostInputImageData);
  free(hostOutputImageData);
>>>>>>> Stashed changes
  return 0;
}

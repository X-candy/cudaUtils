#include "cudaUtility.h"
#include "cudaSubMean.h"

// gpuPreImageNet
__global__ void gpuPreNet( float2 scale, float* input, int iWidth,int iHeight, const float* mean_ptr,float3 mean_shape,float* output, int oWidth, int oHeight )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = oWidth * oHeight;

    if( x >= oWidth || y >= oHeight )
        return;

    //buffer plannar
    //BGR sequence
//    pOutput[ y * oWidth + x] = make_float3(
//            mean_ptr[n * 0 + y * oWidth + x],
//            mean_ptr[n * 1 + y * oWidth + x],
//            mean_ptr[n * 2 + y * oWidth + x]);

//    pOutput[ y * oWidth + x] = make_float3(
//            input[n * 0 + y * oWidth + x] ,
//            input[n * 1 + y * oWidth + x] ,
//            input[n * 2 + y * oWidth + x] );

//    pOutput[ y * oWidth + x] = make_float3(
//            input[n * 0 + y * oWidth + x] - mean_ptr[n * 0 + y * oWidth + x] ,
//            input[n * 1 + y * oWidth + x] - mean_ptr[n * 1 + y * oWidth + x],
//            input[n * 2 + y * oWidth + x] - mean_ptr[n * 2 + y * oWidth + x]);

    //BGR plannar
    output[n * 0 + y * oWidth + x] = input[n * 0 + y * oWidth + x] - mean_ptr[n * 0 + y * oWidth + x];
    output[n * 1 + y * oWidth + x] = input[n * 1 + y * oWidth + x] - mean_ptr[n * 1 + y * oWidth + x];
    output[n * 2 + y * oWidth + x] = input[n * 2 + y * oWidth + x] - mean_ptr[n * 2 + y * oWidth + x];


    //BUFFER img
//    float3* pInput = (float3*)input;

//    pOutput[ y * oWidth + x] = make_float3(
//            pInput[ y * oWidth + x].x -mean_ptr[n * 0 + y * oWidth + x],
//            pInput[ y * oWidth + x].y -mean_ptr[n * 1 + y * oWidth + x],
//            pInput[ y * oWidth + x].z -mean_ptr[n * 2 + y * oWidth + x]);

//    output[n * 0 + y * oWidth + x] = pInput[ y * oWidth + x].x -mean_ptr[n * 0 + y * oWidth + x];
//    output[n * 1 + y * oWidth + x] = pInput[ y * oWidth + x].x -mean_ptr[n * 0 + y * oWidth + x];
//    output[n * 2 + y * oWidth + x] = pInput[ y * oWidth + x].x -mean_ptr[n * 0 + y * oWidth + x];


}


// cudaPreImageNet
cudaError_t cudaPreNet( float* input, size_t inputWidth, size_t inputHeight,const float* mean_ptr,float3 mean_shape,
                        float* output, size_t outputWidth, size_t outputHeight )
{
    if( !input || !output )
        return cudaErrorInvalidDevicePointer;

    if( inputWidth == 0 || outputWidth == 0 || inputHeight == 0 || outputHeight == 0 )
        return cudaErrorInvalidValue;

    const float2 scale = make_float2( float(inputWidth) / float(outputWidth),
                                      float(inputHeight) / float(outputHeight) );

    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth,blockDim.x), iDivUp(outputHeight,blockDim.y));

    gpuPreNet<<<gridDim, blockDim>>>(scale, input, inputWidth,inputHeight, mean_ptr,mean_shape,
            output, outputWidth, outputHeight);


    CUDA(cudaThreadSynchronize());
    return CUDA(cudaGetLastError());
}


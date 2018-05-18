#ifndef __CUDA_SUB_MEAN_H
#define __CUDA_SUB_MEAN_H


#include "cudaUtility.h"
#include <stdint.h>


/**
 * Input image sub mean 
 */

cudaError_t cudaPreNet( float* input,  size_t inputWidth, size_t inputHeight,
                        const float* mean_ptr,float3 mean_shape,
                        float* output, size_t outputWidth, size_t outputHeight );


#endif


#ifndef DLPOOL_NMS_H
#define DLPOOL_NMS_H
#include <vector>
#include <iostream>
#include "cudaUtility.h"
#include "cudaMappedMemory.h"

cudaError_t cudaNMS( float* input_bboxes, size_t nBBoxesCounts,
                     const float nms_overlap_thres,
                     unsigned long long* output_bboxesGPU,
                     std::vector<int> &keep);

#endif //DLPOOL_NMS_H

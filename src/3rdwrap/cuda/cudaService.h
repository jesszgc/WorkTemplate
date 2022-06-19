#ifndef YOLOV5TRTSERVICE_H
#define YOLOV5TRTSERVICE_H
#include "cuda.h"
#include <cuda_runtime.h>
#include<iostream>

namespace akdcuda  {


void* safeCudaMalloc(long memSize);

class cudaService
{
public:
    cudaService();
};












typedef unsigned char uchar ;
__device__ void dev_sigmoid(float *ptr, int len);
__global__ void process_kernel(float *dev_ptr, int height, int width, int no, int total_pix);  // , float* anchors

__global__ void kernel_resize(float *d_dst, int channel,
            int src_h, int src_w,
            int dst_h, int dst_w,
            int top, int bottom, int left, int right,
            uchar *d_src);

void postprocess(float* dev_ptr, int height, int width, int no, int counts);

void mysize(uchar *ptr, float *d_input_tensor,
            int channel,
            int src_h, int src_w,
            int dst_h, int dst_w,
            int top, int bottom, int left, int right);
}
#endif // YOLOV5TRTSERVICE_H

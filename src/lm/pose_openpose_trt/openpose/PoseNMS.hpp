#ifndef POSE_NMS_HPP
#define POSE_NMS_HPP

#include <array>
#include "Point.hpp"


namespace op
{
    // Windows: Cuda functions do not include OP_API
    void nmsGpu(
      float* targetPtr, int* kernelPtr, const float* const sourcePtr, const float threshold, const std::array<int, 4>& targetSize,
      const std::array<int, 4>& sourceSize, const Point<float>& offset);

	void nmsGpuBatch(
		float* targetPtr, int* kernelPtr, const float* sourcePtr, const float threshold, const std::array<int, 4>& targetSize,
		const std::array<int, 4>& sourceSize, const Point<float>& offset, const int batchsize,const int peaksSize, const int kernelSize, const int resizemapSize);
}

#endif // POSE_NMS_HPP
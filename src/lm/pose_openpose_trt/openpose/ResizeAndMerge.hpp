/*
 * @Author: zerollzeng
 * @Date: 2019-10-10 18:07:54
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-10 18:07:54
 */
#ifndef RESIZE_AND_MERGE_HPP
#define RESIZE_AND_MERGE_HPP

#include <vector>
#include <array>


namespace op
{

    // Windows: Cuda functions do not include OP_API
    template <typename T>
    void resizeAndMergeGpu(
        T*targetPtr, const std::vector<const T*>& sourcePtrs, const std::array<int, 4>& targetSize,
        const std::vector<std::array<int, 4>>& sourceSizes, const std::vector<T>& scaleInputToNetInputs = {1.f});

	void resizeAndMergeGpu2Cpu(
		float* targetGpuPtr, float* sourcePtrs, int widthSouce, int heightSource, int channelSource,
		int widthTarget, int heightTarget, int batchsize);

	void resizeAndMergeGpuBatch(
		float* targetGpuPtr, float* sourcePtrs, int widthSouce, int heightSource, int channelSource,
		int widthTarget, int heightTarget, int batchsize);


    // Functions for cvMatToOpInput/cvMatToOpOutput
    template <typename T>
    void resizeAndPadRbgGpu(
        T* targetPtr, const T* const srcPtr, const int sourceWidth, const int sourceHeight,
        const int targetWidth, const int targetHeight, const T scaleFactor);

    template <typename T>
    void resizeAndPadRbgGpu(
        T* targetPtr, const unsigned char* const srcPtr, const int sourceWidth, const int sourceHeight,
        const int targetWidth, const int targetHeight, const T scaleFactor);


	void reorderAndNormalize(
		float* targetPtr, const unsigned char* const srcPtr, const int width, const int height, const int channels);
	void resize8UC3Gpu(const unsigned char*src, int srcWidth, int srcHeight, unsigned char *dst, int dstWidth, int dstHeight);
	void resize8UC332FC3Gpu(const unsigned char*src, int srcWidth, int srcHeight, float *dst, int dstWidth, int dstHeight, float fMean, float fScale);
	void resize8UC332FC3Gpu(const unsigned char*src, int srcWidth, int srcHeight, float *dst, int dstWidth, int dstHeight,float fMean,float fScale, cudaStream_t stream);
}
#endif
// libTrtPose.cpp : ¶¨Òå¿ØÖÆÌ¨Ó¦ÓÃ³ÌÐòµÄÈë¿Úµã¡£
//

#include "libTrtPose.h"
//#include "Caffe2TRT.h"
#include <NvInfer.h>
#include <NvCaffeParser.h>
#include <array>
#include "openpose/ResizeAndMerge.hpp"
#include "openpose/PoseNMS.hpp"
#include "openpose/BodyPartConnector.hpp"
#include "openpose/Point.hpp"
#include "openpose/cuda.cuh"
#include "openpose/cuda.hpp"
#include <cuda.h>
#include "cuda_runtime.h"
#include<time.h>
//#include<omp.h>


libTrtPose::libTrtPose()
{
	bUseMaxGpuMemory = false;
	pVoid = nullptr;
	pStream = nullptr;
	mpKernelGpu = nullptr;
}

libTrtPose::~libTrtPose()
{
	if (pVoid != nullptr)
	{
		Caffe2TRT* pc2trt = (Caffe2TRT*)pVoid;
		delete pc2trt;
		pVoid = nullptr;
	}

	if (pBodyPartPairsGpuPtr != nullptr)
	{
		cudaFree(pBodyPartPairsGpuPtr);
		pBodyPartPairsGpuPtr = nullptr;
	}

	if (pMapIdxGpuPtr != nullptr)
	{
		cudaFree(pMapIdxGpuPtr);
		pMapIdxGpuPtr = nullptr;
	}

	if (pFinalOutputGpuPtr != nullptr)
	{
		cudaFree(pFinalOutputGpuPtr);
		pFinalOutputGpuPtr = nullptr;
	}

	if (pFinalOutputCpuPtr != nullptr)
	{
		delete[] pFinalOutputCpuPtr;
		pFinalOutputCpuPtr = nullptr;
	}

	if (mpKernelGpu != nullptr)
	{
		cudaFree(mpKernelGpu);
		mpKernelGpu = nullptr;
	}

	if (mpPeaksGpu != nullptr)
	{
		cudaFree(mpPeaksGpu);
		mpPeaksGpu = nullptr;
	}

	if (mpPeaksCpu != nullptr)
	{
		delete[] mpPeaksCpu;
		mpPeaksCpu = nullptr;
	}

	if (mpResizeMapGpu != nullptr)
	{
		cudaFree(mpResizeMapGpu);
		mpResizeMapGpu = nullptr;
	}

	if (pInData != nullptr)
	{
		cudaFree(pInData);
		pInData = nullptr;
	}
	if (bUseMaxGpuMemory)
	{
		if (pStream!=nullptr)
		{
			cudaStream_t *streams = (cudaStream_t *)pStream;
			for (int i = 0; i < nMaxBatchSize; i++)
			{
				cudaStreamDestroy(streams[i]);
			}
			pStream = nullptr;
		}
	}
	else
	{
		if (pStream != nullptr)
		{
			cudaStream_t *streams = (cudaStream_t *)pStream;
			for (int i = 0; i < 1; i++)
			{
				cudaStreamDestroy(streams[i]);
			}
			pStream = nullptr;
		}
	}
}

inline void* safeCudaMalloc(size_t memSize)
{
	void* deviceMem = nullptr;
	cudaError cudaerror = cudaMalloc(&deviceMem, memSize);
	if (deviceMem == nullptr || cudaerror != cudaSuccess)
	{
		std::cout << "Out of memory" << std::endl;
	}
	return deviceMem;
}

bool libTrtPose::init(std::string giemodel, int deviceIdx, bool bMaxGpuMem)
{
	int gpuNum;
	cudaGetDeviceCount(&gpuNum);
	if (deviceIdx<gpuNum&&deviceIdx >= 0)
	{
		cudaSetDevice(deviceIdx);
	}
	else
	{
		std::cout << "gpu num: " << gpuNum << std::endl;
		return false;
	}

	Caffe2TRT* pc2trt = new Caffe2TRT;
	std::vector< std::string > OutputNames;
	OutputNames.push_back("net_output");
	std::vector< int > input;
	std::vector< std::vector< int > > output;
	bool binit = pc2trt->initGieModel(giemodel, input, output);
	inputSize = input;
	outputSize = output[0];
	pVoid = pc2trt;
	nMaxBatchSize = pc2trt->nMaxBatchSize;

	mResizeScale = 8;
	int nResizeMapSize = nMaxBatchSize* outputSize[0] * outputSize[1] * mResizeScale * outputSize[2] * mResizeScale * 4;//float 4 sizeof(float)
	mpResizeMapGpu = safeCudaMalloc(nResizeMapSize);

	//mKernelDims = nvinfer1::Dims3(outputSize[0], outputSize[1] * mResizeScale, outputSize[2] * mResizeScale);
	mKernelSize = mNumPeaks * outputSize[1] * mResizeScale * outputSize[2] * mResizeScale;
	mpKernelGpu = safeCudaMalloc(nMaxBatchSize *mKernelSize* 4);

	//mPeaksDims = nvinfer1::Dims3(mNumPeaks, mMaxPerson, mPeaksVector);
	mPeaksSize = mMaxPerson * mNumPeaks * mPeaksVector;
	mpPeaksGpu = safeCudaMalloc(nMaxBatchSize *mPeaksSize * 4);
	mpPeaksCpu = new float[nMaxBatchSize *mPeaksSize];


	auto bodyPartPairs = std::vector<unsigned int>{ 1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24 };
	auto mapIdx = std::vector<unsigned int>{ 0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51 };
	for (int i = 0; i<mapIdx.size(); i++)
	{
		mapIdx[i] += 26;
	}

	pBodyPartPairsGpuPtr = safeCudaMalloc(bodyPartPairs.size() * sizeof(unsigned int));
	cudaMemcpy(pBodyPartPairsGpuPtr, &bodyPartPairs[0], bodyPartPairs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	pMapIdxGpuPtr = safeCudaMalloc(mapIdx.size() * sizeof(unsigned int));
	cudaMemcpy(pMapIdxGpuPtr, &mapIdx[0], mapIdx.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	int numberBodyPartPairs = bodyPartPairs.size() / 2;
	FinalOutputDims.push_back(numberBodyPartPairs);
	FinalOutputDims.push_back(mMaxPerson - 1);
	FinalOutputDims.push_back(mMaxPerson - 1);
	nFinalOutputSize = numberBodyPartPairs*(mMaxPerson - 1)*(mMaxPerson - 1);
	pFinalOutputGpuPtr = safeCudaMalloc(nMaxBatchSize*nFinalOutputSize * sizeof(float));
	pFinalOutputCpuPtr = new float[nMaxBatchSize*nFinalOutputSize];

	nInDataSize = outputSize[1] * outputSize[2] * 3 * sizeof(unsigned char) * 2;
	pInData = safeCudaMalloc(nInDataSize);


	bUseMaxGpuMemory = bMaxGpuMem;
	if (bUseMaxGpuMemory)
	{
		cudaStream_t *streams = (cudaStream_t*)malloc(nMaxBatchSize * sizeof(cudaStream_t));
		for (int i = 0; i < nMaxBatchSize; i++)
		{
			cudaStreamCreate(&(streams[i]));
		}
		pStream = (void*)streams;
	}
	else
	{
		cudaStream_t *streams = (cudaStream_t*)malloc(1 * sizeof(cudaStream_t));
		for (int i = 0; i < 1; i++)
		{
			cudaStreamCreate(&(streams[i]));
		}
		pStream = (void*)streams;
	}


	return binit;
}

bool libTrtPose::init(std::string prototxt, std::string caffemodel, int maxBatchSize, int deviceIdx, bool bMaxGpuMem)
{
	int gpuNum;
	cudaGetDeviceCount(&gpuNum);
	if (deviceIdx<gpuNum&&deviceIdx >= 0)
	{
		cudaSetDevice(deviceIdx);
	}
	else
	{
		std::cout << "gpu num: " << gpuNum << std::endl;
		return false;
	}

	Caffe2TRT* pc2trt = new Caffe2TRT;
	std::vector< std::string > OutputNames;
	OutputNames.push_back("net_output");
	std::vector< int > input;
	std::vector< std::vector< int > > output;
	bool binit = pc2trt->initCaffe(prototxt, caffemodel, OutputNames, maxBatchSize, input, output);
	inputSize = input;
	outputSize = output[0];
	pVoid = pc2trt;
	nMaxBatchSize = maxBatchSize;



	mResizeScale = 8;
	int nResizeMapSize = nMaxBatchSize* outputSize[0] * outputSize[1] * mResizeScale * outputSize[2] * mResizeScale * 4;//float 4 sizeof(float)
	mpResizeMapGpu = safeCudaMalloc(nResizeMapSize);

	//mKernelDims = nvinfer1::Dims3(outputSize[0], outputSize[1] * mResizeScale, outputSize[2] * mResizeScale);
	mKernelSize = mNumPeaks * outputSize[1] * mResizeScale * outputSize[2] * mResizeScale;
	mpKernelGpu = safeCudaMalloc(nMaxBatchSize *mKernelSize * 4);

	//mPeaksDims = nvinfer1::Dims3(mNumPeaks, mMaxPerson, mPeaksVector);

	mPeaksSize = mMaxPerson * mNumPeaks * mPeaksVector;
	mpPeaksGpu = safeCudaMalloc(nMaxBatchSize *mPeaksSize * 4);
	mpPeaksCpu = new float[nMaxBatchSize *mPeaksSize];


	auto bodyPartPairs = std::vector<unsigned int>{ 1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   2,17,  5,18,   14,19,19,20,14,21, 11,22,22,23,11,24 };
	auto mapIdx = std::vector<unsigned int>{ 0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51 };
	for (int i = 0; i<mapIdx.size(); i++)
	{
		mapIdx[i] += 26;
	}

	pBodyPartPairsGpuPtr = safeCudaMalloc(bodyPartPairs.size() * sizeof(unsigned int));
	cudaMemcpy(pBodyPartPairsGpuPtr, &bodyPartPairs[0], bodyPartPairs.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	pMapIdxGpuPtr = safeCudaMalloc(mapIdx.size() * sizeof(unsigned int));
	cudaMemcpy(pMapIdxGpuPtr, &mapIdx[0], mapIdx.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

	int numberBodyPartPairs = bodyPartPairs.size() / 2;
	FinalOutputDims.push_back(numberBodyPartPairs);
	FinalOutputDims.push_back(mMaxPerson - 1);
	FinalOutputDims.push_back(mMaxPerson - 1);
	nFinalOutputSize = numberBodyPartPairs*(mMaxPerson - 1)*(mMaxPerson - 1);
	pFinalOutputGpuPtr = safeCudaMalloc(nMaxBatchSize*nFinalOutputSize * sizeof(float));
	pFinalOutputCpuPtr = new float[nMaxBatchSize*nFinalOutputSize];

	nInDataSize = 0;
	pInData = nullptr;
//	pInData = safeCudaMalloc(nInDataSize);

	bUseMaxGpuMemory = bMaxGpuMem;
	if (bUseMaxGpuMemory)
	{
		cudaStream_t *streams = (cudaStream_t*)malloc(nMaxBatchSize * sizeof(cudaStream_t));
		for (int i = 0; i < nMaxBatchSize; i++)
		{
			cudaStreamCreate(&(streams[i]));
		}
		pStream = (void*)streams;
	}
	else
	{
		cudaStream_t *streams = (cudaStream_t*)malloc(1 * sizeof(cudaStream_t));
		for (int i = 0; i < 1; i++)
		{
			cudaStreamCreate(&(streams[i]));
		}
		pStream = (void*)streams;
	}

	return binit;
}

#define NotUseBatchSize false


bool libTrtPose::infer(std::vector< PoseImg > vecImg, std::vector< std::vector< std::vector< PosePt > > > & outData)
{
	if (vecImg.size()>nMaxBatchSize)
	{
		std::cout << "error:max batchsize ->" << vecImg.size() << std::endl;
		return false;
	}

	Caffe2TRT* pc2trt = (Caffe2TRT*)pVoid;

	void* vecOutMemGpu;
	void* pInNetGpuData = pc2trt->getIndataGpuPtr();

	if (bUseMaxGpuMemory)
	{
		cudaStream_t *streams = (cudaStream_t *)pStream;
		int nMemSize = 0;
		std::vector< int > vecInDataSize;
		for (int i = 0; i < vecImg.size(); i++)
		{
			vecInDataSize.push_back(nMemSize);
			nMemSize += (3 * vecImg[i].w*vecImg[i].h);
		}
		if (nMemSize>nInDataSize)
		{
			if (pInData != nullptr&&nInDataSize != 0)
			{
				cudaFree(pInData);
				pInData = nullptr;
				nInDataSize = 0;
			}
			pInData = safeCudaMalloc(nMemSize * sizeof(unsigned char));
			nInDataSize = nMemSize * sizeof(unsigned char);
		}
		for (int i = 0; i < vecImg.size(); i++)
		{
			int inW = vecImg[i].w;
			int inH = vecImg[i].h;
			unsigned char * pInGpuData = (unsigned char*)pInData + vecInDataSize[i];
			cudaMemcpyAsync(pInGpuData, vecImg[i].data, inW*inH * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice, streams[i]);
			
		}
		for (int i = 0; i < vecImg.size(); i++)
		{
			int outW = inputSize[2];
			int outH = inputSize[1];
			int inW = vecImg[i].w;
			int inH = vecImg[i].h;
			float* pInNetData = (float*)pInNetGpuData;
			pInNetData += (outW*outH * 3 * i);
			op::resize8UC332FC3Gpu((unsigned char*)pInData + vecInDataSize[i], inW, inH, pInNetData, outW, outH, 127.5f, 1.0f / 256.0f, streams[i]);
		}
	}
	else
	{
		for (int i = 0; i < vecImg.size(); i++)
		{
			int outW = inputSize[2];
			int outH = inputSize[1];
			int inW = vecImg[i].w;
			int inH = vecImg[i].h;
			if (inW * inH * 3 * sizeof(unsigned char)>nInDataSize)
			{
				if (pInData!=nullptr&&nInDataSize!=0)
				{
					cudaFree(pInData);
					pInData = nullptr;
					nInDataSize = 0;
				}
				pInData = safeCudaMalloc(inW*inH * 3 * sizeof(unsigned char));
				nInDataSize = inW*inH * 3 * sizeof(unsigned char);
			}
			float* pInNetData = (float*)pInNetGpuData;
			pInNetData += (outW*outH * 3 * i);
			cudaMemcpy(pInData, vecImg[i].data, inW*inH * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			op::resize8UC332FC3Gpu((unsigned char*)pInData, inW, inH, pInNetData, outW, outH, 127.5f, 1.0f / 256.0f);
		}
	}

	clock_t t0 = clock();
	bool binfer = pc2trt->inferGpu_IndataGpu(vecOutMemGpu, vecImg.size());
	clock_t t1 = clock();
	std::cout << "infer: " << t1 - t0<< std::endl;	

	if (binfer)
	{
		int widthSouce = outputSize[2];
		int heightSource = outputSize[1];
		int widthTarget = widthSouce*mResizeScale;
		int heightTarget = heightSource*mResizeScale;
		std::array<int, 4> targetSize2{ 1,mNumPeaks,mMaxPerson,mPeaksVector };
		std::array<int, 4> sourceSize2{ 1,outputSize[0],heightTarget,widthTarget };
		op::Point<float> offset = op::Point<float>(0.5, 0.5);
		op::Point<int> resizeMapSize = op::Point<int>(widthTarget, heightTarget);
		int outW = inputSize[2];
		int outH = inputSize[1];

		//resize
#if NotUseBatchSize
		for (int i = 0; i < vecImg.size(); i++)
		{
			float* pNetOut = (float*)vecOutMemGpu;	pNetOut += (i*outputSize[0] * outputSize[1] * outputSize[2]);
			float* pResizeMap = (float*)mpResizeMapGpu; pResizeMap += (i*outputSize[0] * widthTarget * heightTarget);
			op::resizeAndMergeGpu2Cpu(pResizeMap, pNetOut, widthSouce, heightSource, outputSize[0], widthTarget, heightTarget, 1);
		}
#else
		op::resizeAndMergeGpuBatch((float*)mpResizeMapGpu, (float*)vecOutMemGpu, widthSouce, heightSource, outputSize[0], widthTarget, heightTarget, vecImg.size());
#endif
		// pose nms
#if NotUseBatchSize
		for (int i = 0; i < vecImg.size(); i++)
		{
			float* pResizeMap = (float*)mpResizeMapGpu; pResizeMap += (i*outputSize[0] * widthTarget * heightTarget);
			float* pPeaksGpu = (float*)mpPeaksGpu; pPeaksGpu += (i*mPeaksSize);
			float* pPeaksCpu = (float*)mpPeaksCpu; pPeaksCpu += (i*mPeaksSize);
			int* pKernelGpu = (int*)mpKernelGpu; pKernelGpu += (i*mKernelSize);
			op::nmsGpu(pPeaksGpu, pKernelGpu, pResizeMap, mThreshold, targetSize2, sourceSize2, offset);
			cudaMemcpy(pPeaksCpu, pPeaksGpu, mPeaksSize * 4, cudaMemcpyDeviceToHost);
		}
#else
		op::nmsGpuBatch((float*)mpPeaksGpu, (int*)mpKernelGpu, (float*)mpResizeMapGpu, mThreshold, targetSize2, sourceSize2, offset,vecImg.size(),mPeaksSize,mKernelSize, outputSize[0] * widthTarget * heightTarget);
		//cudaMemcpy(mpPeaksCpu, mpPeaksGpu, mPeaksSize * 4*vecImg.size(), cudaMemcpyDeviceToHost);
		cudaMemcpyAsync(mpPeaksCpu, mpPeaksGpu, mPeaksSize * 4 * vecImg.size(), cudaMemcpyDeviceToHost, ((cudaStream_t*)pStream)[0]);
#endif
		// bodypart connect
#if NotUseBatchSize
		for (int i = 0; i < vecImg.size(); i++)
		{
			float* pResizeMap = (float*)mpResizeMapGpu; pResizeMap += (i*outputSize[0] * widthTarget * heightTarget);
			float* pPeaksGpu = (float*)mpPeaksGpu; pPeaksGpu += (i*mPeaksSize);
			float* pPeaksCpu = (float*)mpPeaksCpu; pPeaksCpu += (i*mPeaksSize);

			float fScaleFactorX = (float)vecImg[i].w / (float)outW;
			float fScaleFactorY = (float)vecImg[i].h / (float)outH;
			std::vector< std::vector< PosePt > > vvPose;
			op::connectBodyPartsGpuLPF(vvPose, pResizeMap, pPeaksCpu, op::PoseModel::BODY_25, resizeMapSize, mMaxPerson - 1, mInterMinAboveThreshold, mInterThreshold, \
				mMinSubsetCnt, mMinSubsetScore, 0.05f, fScaleFactorX, fScaleFactorY, false, (float*)pFinalOutputCpuPtr, FinalOutputDims, (float*)pFinalOutputGpuPtr, \
				(unsigned int*)pBodyPartPairsGpuPtr, (unsigned int*)pMapIdxGpuPtr, pPeaksGpu);
			outData.push_back(vvPose);
		}
#else

		op::connectBodyPartsGpuLPF_Part1((float*)mpResizeMapGpu, outputSize[0] * widthTarget * heightTarget, resizeMapSize, mMaxPerson - 1, mInterMinAboveThreshold, mInterThreshold, \
			 0.05f, (float*)pFinalOutputGpuPtr, nFinalOutputSize,(unsigned int*)pBodyPartPairsGpuPtr, (unsigned int*)pMapIdxGpuPtr, (float*)mpPeaksGpu,mPeaksSize,vecImg.size());
//		cudaMemcpy(pFinalOutputCpuPtr, pFinalOutputGpuPtr, vecImg.size()*nFinalOutputSize * 4, cudaMemcpyDeviceToHost);
		cudaMemcpyAsync(pFinalOutputCpuPtr, pFinalOutputGpuPtr, vecImg.size()*nFinalOutputSize * 4, cudaMemcpyDeviceToHost, ((cudaStream_t*)pStream)[0]);
		cudaStreamSynchronize(((cudaStream_t*)pStream)[0]);
		outData.resize(vecImg.size());
#pragma omp parallel for
		for (int i = 0; i < vecImg.size(); i++)
		{
			float* pPeaksCpu = (float*)mpPeaksCpu; pPeaksCpu += (i*mPeaksSize);

			float fScaleFactorX = (float)vecImg[i].w / (float)outW;
			float fScaleFactorY = (float)vecImg[i].h / (float)outH;
			op::connectBodyPartsGpuLPF_Part2(outData[i], pPeaksCpu, mMaxPerson - 1, mMinSubsetCnt, mMinSubsetScore, \
				fScaleFactorX, fScaleFactorY, false, (float*)pFinalOutputCpuPtr, FinalOutputDims, vecImg.size());
		}
#endif

	}

	return binfer;
}




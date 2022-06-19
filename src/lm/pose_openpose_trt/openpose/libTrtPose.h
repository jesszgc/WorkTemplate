#pragma once
#include<string>
#include<vector>
#include"DataTypes.h"
//#include"NvInfer.h"

class libTrtPose
{
public:
	libTrtPose();
	~libTrtPose();
	
	bool init(std::string prototxt, std::string caffemodel, int maxBatchSize,int deviceIdx,bool bMaxGpuMem);
	bool init(std::string giemodel, int deviceIdx, bool bMaxGpuMem);

	bool infer(std::vector< PoseImg > vecImg, std::vector< std::vector< std::vector< PosePt > > > & outData);


private:
	void* pVoid;
	void* pStream;
	bool bUseMaxGpuMemory;
	int nMaxBatchSize;
	std::vector< int > inputSize;
	std::vector< int > outputSize;

	int nInDataSize;
	void* pInData;

	int mResizeScale;
	void* mpResizeMapGpu;

	void* mpPeaksGpu;
	float* mpPeaksCpu;
	int64_t mPeaksSize;
	const int mNumPeaks = 25;
	int mMaxPerson = 128;
	const int mPeaksVector = 3;

	void* mpKernelGpu;
	int64_t mKernelSize;
	void* pFinalOutputGpuPtr;
	void* pFinalOutputCpuPtr;
	void* pBodyPartPairsGpuPtr;
	void* pMapIdxGpuPtr;
	std::vector<int> FinalOutputDims;
	int nFinalOutputSize;
	// nms parameters
	const float mThreshold = 0.05f;
	const float mNMSoffset = 0.5f;
	// body part connect parameters
	float mInterMinAboveThreshold = 0.95f;
	float mInterThreshold = 0.05f;
	int mMinSubsetCnt = 3;
	float mMinSubsetScore = 0.4f;
	float mScaleFactor = 8.f;
};


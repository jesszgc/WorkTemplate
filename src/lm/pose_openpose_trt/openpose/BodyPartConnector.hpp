/*
 * @Author: zerollzeng
 * @Date: 2019-04-23 14:17:05 
 * @Last Modified by: zerollzeng
 * @Last Modified time: 2019-04-23 19:59:13
 */

#ifndef BODY_PART_CONNECTOR_HPP
#define BODY_PART_CONNECTOR_HPP

#include <vector>
#include <tuple>
#include <iostream>

#include "Array.hpp"
#include "Point.hpp"
#include"../DataTypes.h"

namespace op
{
    enum class PoseModel : unsigned char
    {
        /**
         * COCO + 6 foot keypoints + neck + lower abs model, with 25+1 components (see poseParameters.hpp for details).
         */
        BODY_25 = 0,
        COCO_18,        /**< COCO model + neck, with 18+1 components (see poseParameters.hpp for details). */
        MPI_15,         /**< MPI model, with 15+1 components (see poseParameters.hpp for details). */
        MPI_15_4,       /**< Variation of the MPI model, reduced number of CNN stages to 4: faster but less accurate.*/
        BODY_19,        /**< Experimental. Do not use. */
        BODY_19_X2,     /**< Experimental. Do not use. */
        BODY_19N,       /**< Experimental. Do not use. */
        BODY_25E,       /**< Experimental. Do not use. */
        CAR_12,         /**< Experimental. Do not use. */
        BODY_25D,       /**< Experimental. Do not use. */
        BODY_23,        /**< Experimental. Do not use. */
        CAR_22,         /**< Experimental. Do not use. */
        BODY_19E,       /**< Experimental. Do not use. */
        BODY_25B,       /**< Experimental. Do not use. */
        BODY_135,       /**< Experimental. Do not use. */
        Size,
    };

    void connectBodyPartsCpu(
        Array<float>& poseKeypoints, Array<float>& poseScores, const float* const heatMapPtr, const float* const peaksPtr,
        const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks, const float interMinAboveThreshold,
        const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor = 1.f,
        const bool maximizePositives = false);

    // Windows: Cuda functions do not include OP_API
	void connectBodyPartsGpu(
		Array<float>& poseKeypoints, Array<float>& poseScores, const float* const heatMapGpuPtr, const float* const peaksPtr,
		const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks, const float interMinAboveThreshold,
		const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float defaultNmsThreshold,
		const float scaleFactor, const bool maximizePositives, float* pairScoresCpu, std::vector< int > pairScoresCpuDims,float* pairScoresGpuPtr,
		const unsigned int* const bodyPartPairsGpuPtr, const unsigned int* const mapIdxGpuPtr,
		const float* const peaksGpuPtr);

	void connectBodyPartsGpuLPF(
		std::vector< std::vector< PosePt > > & outData, const float* const heatMapGpuPtr, const float* const peaksPtr,
		const PoseModel poseModel, const Point<int>& heatMapSize, const int maxPeaks, const float interMinAboveThreshold,
		const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float defaultNmsThreshold,
		const float scaleFactorX,const float scaleFactorY, const bool maximizePositives, float* pairScoresCpu, std::vector< int > pairScoresCpuDims, float* pairScoresGpuPtr,
		const unsigned int* const bodyPartPairsGpuPtr, const unsigned int* const mapIdxGpuPtr,
		const float* const peaksGpuPtr);

	void connectBodyPartsGpuLPF_Part1(
		const float* const heatMapGpuPtr,const int resizeMapSize,
		const Point<int>& heatMapSize, const int maxPeaks, const float interMinAboveThreshold,	const float interThreshold, const float defaultNmsThreshold,
		float* pairScoresGpuPtr,int pairScoresSize,
		const unsigned int* const bodyPartPairsGpuPtr, const unsigned int* const mapIdxGpuPtr,
		const float* const peaksGpuPtr,const int peakSize,const int batchsize);

	void connectBodyPartsGpuLPF_Part2(
		std::vector< std::vector< PosePt > > & outData, const float* const peaksPtr,	const int maxPeaks,	const int minSubsetCnt, const float minSubsetScore, 
		const float scaleFactorX, const float scaleFactorY, const bool maximizePositives, float* pairScoresCpu, std::vector< int > pairScoresCpuDims, const int batchsize);


    // Private functions used by the 2 above functions
    template <typename T>
    std::vector<std::pair<std::vector<int>, T>> createPeopleVector(
        const T* const heatMapPtr, const T* const peaksPtr, const PoseModel poseModel, const Point<int>& heatMapSize,
        const int maxPeaks, const T interThreshold, const T interMinAboveThreshold,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs, const Array<T>& precomputedPAFs = Array<T>());

    template <typename T>
    void removePeopleBelowThresholdsAndFillFaces(
        std::vector<int>& validSubsetIndexes, int& numberPeople,
        std::vector<std::pair<std::vector<int>, T>>& subsets, const unsigned int numberBodyParts,
        const int minSubsetCnt, const T minSubsetScore, const bool maximizePositives, const T* const peaksPtr);

    template <typename T>
    void peopleVectorToPeopleArray(
        Array<T>& poseKeypoints, Array<T>& poseScores, const T scaleFactor,
        const std::vector<std::pair<std::vector<int>, T>>& subsets, const std::vector<int>& validSubsetIndexes,
        const T* const peaksPtr, const int numberPeople, const unsigned int numberBodyParts,
        const unsigned int numberBodyPartPairs);

    template <typename T>
    std::vector<std::tuple<T, T, int, int, int>> pafPtrIntoVector(
		const T* pairScores, std::vector< int > pairScoresDims, const T* const peaksPtr, const int maxPeaks,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyPartPairs);

    template <typename T>
    std::vector<std::pair<std::vector<int>, T>> pafVectorIntoPeopleVector(
        const std::vector<std::tuple<T, T, int, int, int>>& pairScores, const T* const peaksPtr, const int maxPeaks,
        const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts);
}


#endif
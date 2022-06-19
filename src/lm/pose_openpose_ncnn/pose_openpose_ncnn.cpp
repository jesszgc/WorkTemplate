#include"pose_openpose_ncnn.h"
#include "layer.h"
#include "net.h"
#include <float.h>
#include <stdio.h>
#include <vector>
#include<iostream>
#include"opencv2/opencv.hpp"
#define MAX_STRIDE 32
poseOpenposeNcnn::poseOpenposeNcnn(std::string modelpath,int gpu,int threadnum,int modelsize,int maxbatchsize)
{
    ncnn::Net *yolov5=new ncnn::Net();

    if(gpu>-1) {
        yolov5->opt.use_vulkan_compute = true;
    }
    //yolov5.opt.use_bf16_storage = true;
    yolov5->opt.num_threads=2;
    std::string parampath=modelpath+".param";
    std::string binpath=modelpath+".bin";

    int ret1=yolov5->load_param(parampath.c_str());
    int ret2=yolov5->load_model(binpath.c_str());
    if(ret1>-1&&ret2>-1)
    {
        initsuccess=1;
        net=yolov5;
    } else
    {
        initsuccess=-1;
    }
   // std::cout<<"ret1,ret2:"<<ret1<<"__"<<ret2<<std::endl;
    // #endif
}
poseOpenposeNcnn::~poseOpenposeNcnn(){
    if(net!=nullptr)
    {
        delete (ncnn::Net*)net;
        net=nullptr;
    }
}
//const std::vector<unsigned int> POSE_COCO_PAIRS{ 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17 };
//const std::vector<unsigned int> POSE_COCO_MAP_IDX{ 31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46 };
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <functional>
//template<typename T>
//inline int intRound(const T a)
//{
//    return int(a+0.5f);
//}
//// Max/min functions
//template<typename T>
//inline T fastMax(const T a, const T b)
//{
//    return (a > b ? a : b);
//}

//template<typename T>
//inline T fastMin(const T a, const T b)
//{
//    return (a < b ? a : b);
//}
const unsigned int POSE_MAX_PEOPLE = 96;
////根据得到的结果，连接身体区域
//void connectBodyPartsCpu(std::vector<float>& poseKeypoints, const float* const heatMapPtr, const float* const peaksPtr,
//    const cv::Size& heatMapSize, const int maxPeaks, const int interMinAboveThreshold,
//    const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor, std::vector<int>& keypointShape)
//{
//    keypointShape.resize(3);
//   const auto& bodyPartPairs = POSE_COCO_PAIRS;
//    const auto& mapIdx = POSE_COCO_MAP_IDX;
//    const auto numberBodyParts = 18;

//    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

//    std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
//    const auto subsetCounterIndex = numberBodyParts;
//    const auto subsetSize = numberBodyParts + 1;

//    const auto peaksOffset = 3 * (maxPeaks + 1);
//    const auto heatMapOffset = heatMapSize.area();

//    for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
//    {
//        const auto bodyPartA = bodyPartPairs[2 * pairIndex];
//        const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
//        const auto* candidateA = peaksPtr + bodyPartA*peaksOffset;
//        const auto* candidateB = peaksPtr + bodyPartB*peaksOffset;
//        const auto nA = intRound(candidateA[0]);
//        const auto nB = intRound(candidateB[0]);

//        // add parts into the subset in special case
//        if (nA == 0 || nB == 0)
//        {
//            // Change w.r.t. other
//            if (nA == 0) // nB == 0 or not
//            {
//                for (auto i = 1; i <= nB; i++)
//                {
//                    bool num = false;
//                    const auto indexB = bodyPartB;
//                    for (auto j = 0u; j < subset.size(); j++)
//                    {
//                        const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
//                        if (subset[j].first[indexB] == off)
//                        {
//                            num = true;
//                            break;
//                        }
//                    }
//                    if (!num)
//                    {
//                        std::vector<int> rowVector(subsetSize, 0);
//                        rowVector[bodyPartB] = bodyPartB*peaksOffset + i * 3 + 2; //store the index
//                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
//                        const auto subsetScore = candidateB[i * 3 + 2]; //second last number in each row is the total score
//                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
//                    }
//                }
//            }
//            else // if (nA != 0 && nB == 0)
//            {
//                for (auto i = 1; i <= nA; i++)
//                {
//                    bool num = false;
//                    const auto indexA = bodyPartA;
//                    for (auto j = 0u; j < subset.size(); j++)
//                    {
//                        const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
//                        if (subset[j].first[indexA] == off)
//                        {
//                            num = true;
//                            break;
//                        }
//                    }
//                    if (!num)
//                    {
//                        std::vector<int> rowVector(subsetSize, 0);
//                        rowVector[bodyPartA] = bodyPartA*peaksOffset + i * 3 + 2; //store the index
//                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
//                        const auto subsetScore = candidateA[i * 3 + 2]; //second last number in each row is the total score
//                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
//                    }
//                }
//            }
//        }
//        else // if (nA != 0 && nB != 0)
//        {
//            std::vector<std::tuple<double, int, int>> temp;
//            const auto numInter = 10;
//            const auto* const mapX = heatMapPtr + mapIdx[2 * pairIndex] * heatMapOffset;
//            const auto* const mapY = heatMapPtr + mapIdx[2 * pairIndex + 1] * heatMapOffset;
//            for (auto i = 1; i <= nA; i++)
//            {
//                for (auto j = 1; j <= nB; j++)
//                {
//                    const auto dX = candidateB[j * 3] - candidateA[i * 3];
//                    const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
//                    const auto normVec = float(std::sqrt(dX*dX + dY*dY));
//                    // If the peaksPtr are coincident. Don't connect them.
//                    if (normVec > 1e-6)
//                    {
//                        const auto sX = candidateA[i * 3];
//                        const auto sY = candidateA[i * 3 + 1];
//                        const auto vecX = dX / normVec;
//                        const auto vecY = dY / normVec;

//                        auto sum = 0.;
//                        auto count = 0;
//                        for (auto lm = 0; lm < numInter; lm++)
//                        {
//                            const auto mX = fastMin(heatMapSize.width - 1, intRound(sX + lm*dX / numInter));
//                            const auto mY = fastMin(heatMapSize.height - 1, intRound(sY + lm*dY / numInter));
//                            //checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
//                            //checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
//                            const auto idx = mY * heatMapSize.width + mX;
//                            const auto score = (vecX*mapX[idx] + vecY*mapY[idx]);
//                            if (score > interThreshold)
//                            {
//                                sum += score;
//                                count++;
//                            }
//                        }

//                        // parts score + connection score
//                        if (count > interMinAboveThreshold)
//                            temp.emplace_back(std::make_tuple(sum / count, i, j));
//                    }
//                }
//            }

//            // select the top minAB connection, assuming that each part occur only once
//            // sort rows in descending order based on parts + connection score
//            if (!temp.empty())
//                std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

//            std::vector<std::tuple<int, int, double>> connectionK;

//            const auto minAB = fastMin(nA, nB);
//            std::vector<int> occurA(nA, 0);
//            std::vector<int> occurB(nB, 0);
//            auto counter = 0;
//            for (auto row = 0u; row < temp.size(); row++)
//            {
//                const auto score = std::get<0>(temp[row]);
//                const auto x = std::get<1>(temp[row]);
//                const auto y = std::get<2>(temp[row]);
//                if (!occurA[x - 1] && !occurB[y - 1])
//                {
//                    connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
//                        bodyPartB*peaksOffset + y * 3 + 2,
//                        score));
//                    counter++;
//                    if (counter == minAB)
//                        break;
//                    occurA[x - 1] = 1;
//                    occurB[y - 1] = 1;
//                }
//            }

//            // Cluster all the body part candidates into subset based on the part connection
//            // initialize first body part connection 15&16
//            if (pairIndex == 0)
//            {
//                for (const auto connectionKI : connectionK)
//                {
//                    std::vector<int> rowVector(numberBodyParts + 3, 0);
//                    const auto indexA = std::get<0>(connectionKI);
//                    const auto indexB = std::get<1>(connectionKI);
//                    const auto score = std::get<2>(connectionKI);
//                    rowVector[bodyPartPairs[0]] = indexA;
//                    rowVector[bodyPartPairs[1]] = indexB;
//                    rowVector[subsetCounterIndex] = 2;
//                    // add the score of parts and the connection
//                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
//                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
//                }
//            }
//            // Add ears connections (in case person is looking to opposite direction to camera)
//            else if (pairIndex == 17 || pairIndex == 18)
//            {
//                for (const auto& connectionKI : connectionK)
//                {
//                    const auto indexA = std::get<0>(connectionKI);
//                    const auto indexB = std::get<1>(connectionKI);
//                    for (auto& subsetJ : subset)
//                    {
//                        auto& subsetJFirst = subsetJ.first[bodyPartA];
//                        auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
//                        if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
//                            subsetJFirstPlus1 = indexB;
//                        else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
//                            subsetJFirst = indexA;
//                    }
//                }
//            }
//            else
//            {
//                if (!connectionK.empty())
//                {
//                    // A is already in the subset, find its connection B
//                    for (auto i = 0u; i < connectionK.size(); i++)
//                    {
//                        const auto indexA = std::get<0>(connectionK[i]);
//                        const auto indexB = std::get<1>(connectionK[i]);
//                        const auto score = std::get<2>(connectionK[i]);
//                        auto num = 0;
//                        for (auto j = 0u; j < subset.size(); j++)
//                        {
//                            if (subset[j].first[bodyPartA] == indexA)
//                            {
//                                subset[j].first[bodyPartB] = indexB;
//                                num++;
//                                subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
//                                subset[j].second = subset[j].second + peaksPtr[indexB] + score;
//                            }
//                        }
//                        // if can not find partA in the subset, create a new subset
//                        if (num == 0)
//                        {
//                            std::vector<int> rowVector(subsetSize, 0);
//                            rowVector[bodyPartA] = indexA;
//                            rowVector[bodyPartB] = indexB;
//                            rowVector[subsetCounterIndex] = 2;
//                            const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
//                            subset.emplace_back(std::make_pair(rowVector, subsetScore));
//                        }
//                    }
//                }
//            }
//        }
//    }

//    // Delete people below the following thresholds:
//    // a) minSubsetCnt: removed if less than minSubsetCnt body parts
//    // b) minSubsetScore: removed if global score smaller than this
//    // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
//    auto numberPeople = 0;
//    std::vector<int> validSubsetIndexes;
//    validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
//    for (auto index = 0u; index < subset.size(); index++)
//    {
//        const auto subsetCounter = subset[index].first[subsetCounterIndex];
//        const auto subsetScore = subset[index].second;
//        if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
//        {
//            numberPeople++;
//            validSubsetIndexes.emplace_back(index);
//            if (numberPeople == POSE_MAX_PEOPLE)
//                break;
//        }
//        else if (subsetCounter < 1)
//            printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
//    }

//    // Fill and return poseKeypoints
//    keypointShape = { numberPeople, (int)numberBodyParts, 3 };
//    if (numberPeople > 0)
//        poseKeypoints.resize(numberPeople * (int)numberBodyParts * 3);
//    else
//        poseKeypoints.clear();

//    for (auto person = 0u; person < validSubsetIndexes.size(); person++)
//    {
//        const auto& subsetI = subset[validSubsetIndexes[person]].first;
//        for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
//        {
//            const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
//            const auto bodyPartIndex = subsetI[bodyPart];
//            if (bodyPartIndex > 0)
//            {
//                poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
//                poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
//                poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
//            }
//            else
//            {
//                poseKeypoints[baseOffset] = 0.f;
//                poseKeypoints[baseOffset + 1] = 0.f;
//                poseKeypoints[baseOffset + 2] = 0.f;
//            }
//        }
//    }
//}
////bottom_blob是输入，top是输出
//void nms( float* bottom_blob,int w,int h, std::vector<float> &peaks, float threshold){
//    //maxPeaks就是最大人数，+1是为了第一位存个数
//    //算法，是每个点，如果大于阈值，同时大于上下左右值的时候，则认为是峰值

//    //算法很简单，featuremap的任意一个点，其上下左右和斜上下左右，都小于自身，就认为是要的点
//    //然后以该点区域，选择7*7区域，按照得分值和x、y来计算最合适的亚像素坐标

//    float* ptr = bottom_blob;
//    int num_peaks = 0;
//    for (int y = 1; y < h - 1 ; ++y){
//        for (int x = 1; x < w - 1; ++x){

//            float value = ptr[y*w + x];
//            if (value <= threshold){
//                continue;
//            }
//            const float topLeft = ptr[(y - 1)*w + x - 1];
//            const float top = ptr[(y - 1)*w + x];
//            const float topRight = ptr[(y - 1)*w + x + 1];
//            const float left = ptr[y*w + x - 1];
//            const float right = ptr[y*w + x + 1];
//            const float bottomLeft = ptr[(y + 1)*w + x - 1];
//            const float bottom = ptr[(y + 1)*w + x];
//            const float bottomRight = ptr[(y + 1)*w + x + 1];

//            if (value <= topLeft || value <= top || value <= topRight
//                    ||  value <= left ||  value <= right
//                    ||  value <= bottomLeft ||  value <= bottom ||  value <= bottomRight)
//            {
//                continue;
//            }
//            //计算亚像素坐标
//            float xAcc = 0;
//            float yAcc = 0;
//            float scoreAcc = 0;
//            for (int kx = -3; kx <= 3; ++kx){
//                int ux = x + kx;
//                if (ux >= 0 && ux < w){
//                    for (int ky = -3; ky <= 3; ++ky){
//                        int uy = y + ky;
//                        if (uy >= 0 && uy < h){
//                            float score = ptr[uy * w + ux];
//                            xAcc += ux * score;
//                            yAcc += uy * score;
//                            scoreAcc += score;
//                        }
//                    }
//                }
//            }

//            xAcc /= scoreAcc;
//            yAcc /= scoreAcc;
//            scoreAcc = value;
//            peaks.push_back(xAcc);
//            peaks.push_back(yAcc);
//            peaks.push_back(scoreAcc);


//        }
//    }
//}



int genoutput(ncnn::Mat &data,inferOutdata &outdata)
{
    std::vector<std::vector<float>> pps;
    for(int i=0;i<data.c;++i)
    {
        const ncnn::Mat feat = data.channel(i);
        std::vector<float> peaks;
        //nms((float*)feat.data,feat.w,feat.h,peaks,0.1);
        pps.push_back(peaks);
    }

    return -1;
}



//const std::map<unsigned int, std::string> POSE_BODY_25B_BODY_PARTS{
//	{ 0,  "Nose" },
//	{ 1,  "LEye" },
//	{ 2,  "REye" },
//	{ 3,  "LEar" },
//	{ 4,  "REar" },
//	{ 5,  "LShoulder" },
//	{ 6,  "RShoulder" },
//	{ 7,  "LElbow" },
//	{ 8,  "RElbow" },
//	{ 9,  "LWrist" },
//	{ 10, "RWrist" },
//	{ 11, "LHip" },
//	{ 12, "RHip" },
//	{ 13, "LKnee" },
//	{ 14, "RKnee" },
//	{ 15, "LAnkle" },
//	{ 16, "RAnkle" },
//	{ 17, "UpperNeck" },
//	{ 18, "HeadTop" },
//	{ 19, "LBigToe" },
//	{ 20, "LSmallToe" },
//	{ 21, "LHeel" },
//	{ 22, "RBigToe" },
//	{ 23, "RSmallToe" },
//	{ 24, "RHeel" },
//};
//std::vector<unsigned int> POSE_MAP_INDEX = std::vector<unsigned int>{
//	// Minimum spanning tree
//	// |------------------------------------------- COCO Body -------------------------------------------|
//	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
//	// Redundant ones
//	// |------------------ Foot ------------------| |-- MPII --|
//	32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
//	// Redundant ones
//	// MPII redundant, ears, ears-shoulders, shoulders-wrists, wrists, wrists-hips, hips, ankles)
//	48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71
//};
//	// BODY_135
#define POSE_MAX_PEOPLE 96
//body25
//const std::vector<unsigned int> POSE_BODY25_PAIRS{ 1,8,   1,2,   1,5,   2,3,   3,4,   5,6,   6,7,   8,9,   9,10,  10,11, 8,12,  12,13, 13,14,  1,0,   0,15, 15,17,  0,16, 16,18,   14,19,19,20,14,21, 11,22,22,23,11,24 };
//const std::vector<unsigned int> POSE_BODY25_MAP_IDX{ 0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51 };

//body25b
const std::vector<unsigned int>POSE_BODY25B_PAIRS{ 0, 1, 0, 2, 1, 3, 2, 4, 0, 5, 0, 6, 5, 7, 6, 8, 7, 9, 8, 10, 5, 11, 6, 12, 11, 13, 12, 14, 13, 15, 14, 16,15, 19, 19, 20, 15, 21, 16, 22, 22, 23, 16, 24, 5, 17, 5, 18,6, 17, 6, 18, 3, 4, 3, 5, 4, 6, 5, 9, 6, 10, 9, 10, 9, 11, 10, 12, 11, 12, 15, 16 };
//const std::vector<unsigned int>POSE_BODY25B_MAP_IDX{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71 };
const std::vector<unsigned int> POSE_BODY25B_MAP_IDX{ 26,27, 40,41, 48,49, 42,43, 44,45, 50,51, 52,53, 32,33, 28,29, 30,31, 34,35, 36,37, 38,39, 56,57, 58,59, 62,63, 60,61, 64,65, 46,47, 54,55, 66,67,68,69,70,71, 72,73,74,75,76,77 };

//const std::vector<unsigned int>POSE_COCO_MAP_IDX{ 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96 };

//bottom_blob是输入，top是输出
void nms(ncnn::Mat bottom_blob, ncnn::Mat& top_blob, float threshold)
{
    //maxPeaks就是最大人数，+1是为了第一位存个数
    //算法，是每个点，如果大于阈值，同时大于上下左右值的时候，则认为是峰值

    //算法很简单，featuremap的任意一个点，其上下左右和斜上下左右，都小于自身，就认为是要的点
    //然后以该点区域，选择7*7区域，按照得分值和x、y来计算最合适的亚像素坐标

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int plane_offset = w * h;
    float* ptr = (float*)bottom_blob.data;
    float* top_ptr = (float*)top_blob.data;
    int top_plane_offset = top_blob.w * top_blob.h;
    int max_peaks = top_blob.h - 1;

    //for (int n = 0; n < bottom_blob->; ++n)
    {
        for (int c = 0; c < bottom_blob.c - 1; ++c)
        {

            int num_peaks = 0;
            for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y)
            {
                for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x)
                {
                    float value = ptr[y*w + x];
                    if (value > threshold) {
                        const float topLeft = ptr[(y - 1)*w + x - 1];
                        const float top = ptr[(y - 1)*w + x];
                        const float topRight = ptr[(y - 1)*w + x + 1];
                        const float left = ptr[y*w + x - 1];
                        const float right = ptr[y*w + x + 1];
                        const float bottomLeft = ptr[(y + 1)*w + x - 1];
                        const float bottom = ptr[(y + 1)*w + x];
                        const float bottomRight = ptr[(y + 1)*w + x + 1];

                        if (value > topLeft && value > top && value > topRight
                            && value > left && value > right
                            && value > bottomLeft && value > bottom && value > bottomRight)
                        {
                            //计算亚像素坐标
                            float xAcc = 0;
                            float yAcc = 0;
                            float scoreAcc = 0;
                            for (int kx = -3; kx <= 3; ++kx) {
                                int ux = x + kx;
                                if (ux >= 0 && ux < w) {
                                    for (int ky = -3; ky <= 3; ++ky) {
                                        int uy = y + ky;
                                        if (uy >= 0 && uy < h) {
                                            float score = ptr[uy * w + ux];
                                            xAcc += ux * score;
                                            yAcc += uy * score;
                                            scoreAcc += score;
                                        }
                                    }
                                }
                            }

                            xAcc /= scoreAcc;
                            yAcc /= scoreAcc;
                            scoreAcc = value;
                            top_ptr[(num_peaks + 1) * 3 + 0] = xAcc;
                            top_ptr[(num_peaks + 1) * 3 + 1] = yAcc;
                            top_ptr[(num_peaks + 1) * 3 + 2] = scoreAcc;
                            num_peaks++;
                        }
                    }
                }
            }
            top_ptr[0] = num_peaks;
            ptr += plane_offset;
            top_ptr += top_plane_offset;
        }
    }
}


/*---------版本2--------*/
inline float getScoreAB(const int i, const int j, const float* const candidateAPtr, const float* const candidateBPtr,
    const float* const mapX, const float* const mapY, const int & heatMapWidth, const int & heatMapHeight,
    const float interThreshold, const int interMinAboveThreshold)
{
    const auto vectorAToBX = candidateBPtr[j * 3] - candidateAPtr[i * 3];
    const auto vectorAToBY = candidateBPtr[j * 3 + 1] - candidateAPtr[i * 3 + 1];
    const auto vectorAToBMax = std::abs(vectorAToBX)>std::abs(vectorAToBY) ? std::abs(vectorAToBX) : std::abs(vectorAToBY);

    int nTempmin = 25 < (int)(std::sqrt(5 * vectorAToBMax) + 0.5f) ? 25 : (int)(std::sqrt(5 * vectorAToBMax) + 0.5f);
    const auto numberPointsInLine = 5 > nTempmin ? 5 : nTempmin;

    const auto vectorNorm = (float)(std::sqrt(vectorAToBX*vectorAToBX + vectorAToBY*vectorAToBY));
    // If the peaksPtr are coincident. Don't connect them.
    if (vectorNorm > 1e-6)
    {
        const auto sX = candidateAPtr[i * 3];
        const auto sY = candidateAPtr[i * 3 + 1];
        const auto vectorAToBNormX = vectorAToBX / vectorNorm;
        const auto vectorAToBNormY = vectorAToBY / vectorNorm;

        float sum = 0.0f;
        auto count = 0u;
        const auto vectorAToBXInLine = vectorAToBX / numberPointsInLine;
        const auto vectorAToBYInLine = vectorAToBY / numberPointsInLine;
        for (auto lm = 0; lm < numberPointsInLine; lm++)
        {
            const auto mX = std::max(
                0, std::min(heatMapWidth - 1, (int)(sX + lm*vectorAToBXInLine + 0.5f)));
            const auto mY = std::max(
                0, std::min(heatMapHeight - 1, (int)(sY + lm*vectorAToBYInLine + 0.5f)));
            const auto idx = mY * heatMapWidth + mX;
            const auto score = (vectorAToBNormX*mapX[idx] + vectorAToBNormY*mapY[idx]);
            if (score > interThreshold)
            {
                sum += score;
                count++;
            }
        }
        if (count / (float)numberPointsInLine > interMinAboveThreshold)
            return sum / (count + 0.0000001f);
    }
    return 0.0f;
}

std::vector<std::pair<std::vector<int>, double>> generateInitialSubsets(
    const float* const heatMapPtr, const float* const peaksPtr, const int & heatMapWidth, const int & heatMapHeight,
    const int maxPeaks, const float interThreshold, const int interMinAboveThreshold,
    const std::vector<unsigned int>& bodyPartPairs, const unsigned int numberBodyParts,
    const unsigned int numberBodyPartPairs, const unsigned int subsetCounterIndex)
{
    // std::vector<std::pair<std::vector<int>, double>> refers to:
    //     - std::vector<int>: [body parts locations, #body parts found]
    //     - double: subset score
    std::vector<std::pair<std::vector<int>, double>> subsets;

    const auto& mapIdx = POSE_BODY25B_MAP_IDX;
    const auto numberBodyPartsAndBkg = numberBodyParts;
    const auto subsetSize = numberBodyParts;
    const auto peaksOffset = 3 * (maxPeaks + 1);
    const auto heatMapOffset = heatMapWidth*heatMapHeight;
    // Star-PAF
    const auto bodyPart0 = 1;
    const auto* candidate0Ptr = peaksPtr + bodyPart0*peaksOffset;
    const auto number0 = (int)(candidate0Ptr[0] + 0.5f);
    // Iterate over it PAF connection, e.g. neck-nose, neck-Lshoulder, etc.
    for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
    {
        const auto bodyPartA = bodyPartPairs[2 * pairIndex];
        const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
        const auto* candidateAPtr = peaksPtr + bodyPartA*peaksOffset;
        const auto* candidateBPtr = peaksPtr + bodyPartB*peaksOffset;
        const auto numberA = (int)(candidateAPtr[0] + 0.5f);
        const auto numberB = (int)(candidateBPtr[0] + 0.5f);

        // E.g. neck-nose connection. If one of them is empty (e.g. no noses detected)
        // Add the non-empty elements into the subsets
        if (numberA == 0 || numberB == 0)
        {
            // E.g. neck-nose connection. If no necks, add all noses
            // Change w.r.t. other
            if (numberA == 0) // numberB == 0 or not
            {
                // Non-MPI
                for (auto i = 1; i <= numberB; i++)
                {
                    bool found = false;
                    for (const auto& subset : subsets)
                    {
                        const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
                        if (subset.first[bodyPartB] == off)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        // Store the index
                        rowVector[bodyPartB] = bodyPartB*peaksOffset + i * 3 + 2;
                        // Last number in each row is the parts number of that person
                        rowVector[subsetCounterIndex] = 1;
                        const auto subsetScore = candidateBPtr[i * 3 + 2];
                        // Second last number in each row is the total score
                        subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
            // E.g. neck-nose connection. If no noses, add all necks
            else // if (numberA != 0 && numberB == 0)
            {
                // Non-MPI
                for (auto i = 1; i <= numberA; i++)
                {
                    bool found = false;
                    const auto indexA = bodyPartA;
                    for (const auto& subset : subsets)
                    {
                        const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
                        if (subset.first[indexA] == off)
                        {
                            found = true;
                            break;
                        }
                    }
                    if (!found)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        // Store the index
                        rowVector[bodyPartA] = bodyPartA*peaksOffset + i * 3 + 2;
                        // Last number in each row is the parts number of that person
                        rowVector[subsetCounterIndex] = 1;
                        // Second last number in each row is the total score
                        const auto subsetScore = candidateAPtr[i * 3 + 2];
                        subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
        }
        // E.g. neck-nose connection. If necks and noses, look for maximums
        else // if (numberA != 0 && numberB != 0)
        {
            // (score, x, y). Inverted order for easy std::sort
            std::vector<std::tuple<double, int, int>> allABConnections;
            // Note: Problem of this function, if no right PAF between A and B, both elements are discarded.
            // However, they should be added indepently, not discarded
            {
                const auto* mapX = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2 * pairIndex]) * heatMapOffset;
                const auto* mapY = heatMapPtr + (numberBodyPartsAndBkg + mapIdx[2 * pairIndex + 1]) * heatMapOffset;
                // E.g. neck-nose connection. For each neck
                for (auto i = 1; i <= numberA; i++)
                {
                    // E.g. neck-nose connection. For each nose
                    for (auto j = 1; j <= numberB; j++)
                    {
                        // Initial PAF
                        auto scoreAB = getScoreAB(i, j, candidateAPtr, candidateBPtr, mapX, mapY,
                            heatMapWidth, heatMapHeight, interThreshold, interMinAboveThreshold);

                        // E.g. neck-nose connection. If possible PAF between neck i, nose j --> add
                        // parts score + connection score
                        if (scoreAB > 1e-6)
                            allABConnections.emplace_back(std::make_tuple(scoreAB, i, j));
                    }
                }
            }

            // select the top minAB connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (!allABConnections.empty())
                std::sort(allABConnections.begin(), allABConnections.end(),
                    std::greater<std::tuple<double, int, int>>());

            std::vector<std::tuple<int, int, double>> abConnections; // (x, y, score)
            {
                const auto minAB = std::min(numberA, numberB);
                std::vector<int> occurA(numberA, 0);
                std::vector<int> occurB(numberB, 0);
                auto counter = 0;
                for (auto row = 0u; row < allABConnections.size(); row++)
                {
                    const auto score = std::get<0>(allABConnections[row]);
                    const auto i = std::get<1>(allABConnections[row]);
                    const auto j = std::get<2>(allABConnections[row]);
                    if (!occurA[i - 1] && !occurB[j - 1])
                    {
                        abConnections.emplace_back(std::make_tuple(bodyPartA*peaksOffset + i * 3 + 2,
                            bodyPartB*peaksOffset + j * 3 + 2,
                            score));
                        counter++;
                        if (counter == minAB)
                            break;
                        occurA[i - 1] = 1;
                        occurB[j - 1] = 1;
                    }
                }
            }

            // Cluster all the body part candidates into subsets based on the part connection
            if (!abConnections.empty())
            {
                // initialize first body part connection 15&16
                if (pairIndex == 0)
                {
                    for (const auto& abConnection : abConnections)
                    {
                        std::vector<int> rowVector(numberBodyParts + 3, 0);
                        const auto indexA = std::get<0>(abConnection);
                        const auto indexB = std::get<1>(abConnection);
                        const auto score = std::get<2>(abConnection);
                        rowVector[bodyPartPairs[0]] = indexA;
                        rowVector[bodyPartPairs[1]] = indexB;
                        rowVector[subsetCounterIndex] = 2;
                        // add the score of parts and the connection
                        const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                        subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
                // Add ears connections (in case person is looking to opposite direction to camera)
                // Note: This has some issues:
                //     - It does not prevent repeating the same keypoint in different people
                //     - Assuming I have nose,eye,ear as 1 subset, and whole arm as another one, it will not
                //       merge them both
                else if (numberBodyParts == 25 && (pairIndex == 18 || pairIndex == 19))
                {
                    for (const auto& abConnection : abConnections)
                    {
                        const auto indexA = std::get<0>(abConnection);
                        const auto indexB = std::get<1>(abConnection);
                        for (auto& subset : subsets)
                        {
                            auto& subsetA = subset.first[bodyPartA];
                            auto& subsetB = subset.first[bodyPartB];
                            if (subsetA == indexA && subsetB == 0)
                            {
                                subsetB = indexB;
                                // // This seems to harm acc 0.1% for BODY_25
                                // subset.first[subsetCounterIndex]++;
                            }
                            else if (subsetB == indexB && subsetA == 0)
                            {
                                subsetA = indexA;
                                // // This seems to harm acc 0.1% for BODY_25
                                // subset.first[subsetCounterIndex]++;
                            }
                        }
                    }
                }
                else
                {
                    // A is already in the subsets, find its connection B
                    for (const auto& abConnection : abConnections)
                    {
                        const auto indexA = std::get<0>(abConnection);
                        const auto indexB = std::get<1>(abConnection);
                        const auto score = std::get<2>(abConnection);
                        bool found = false;
                        for (auto& subset : subsets)
                        {
                            // Found partA in a subsets, add partB to same one.
                            if (subset.first[bodyPartA] == indexA)
                            {
                                subset.first[bodyPartB] = indexB;
                                subset.first[subsetCounterIndex]++;
                                subset.second += peaksPtr[indexB] + score;
                                found = true;
                                break;
                            }
                        }
                        // Not found partA in subsets, add new subsets element
                        if (!found)
                        {
                            std::vector<int> rowVector(subsetSize, 0);
                            rowVector[bodyPartA] = indexA;
                            rowVector[bodyPartB] = indexB;
                            rowVector[subsetCounterIndex] = 2;
                            const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                            subsets.emplace_back(std::make_pair(rowVector, subsetScore));
                        }
                    }
                }
            }
        }
    }
    return subsets;
}

void removeSubsetsBelowThresholds(std::vector<int>& validSubsetIndexes, int& numberPeople,
    const std::vector<std::pair<std::vector<int>, double>>& subsets,
    const unsigned int subsetCounterIndex, const unsigned int numberBodyParts,
    const int minSubsetCnt, const float minSubsetScore, const int maxPeaks)
{
    // Delete people below the following thresholds:
    // a) minSubsetCnt: removed if less than minSubsetCnt body parts
    // b) minSubsetScore: removed if global score smaller than this
    // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
    numberPeople = 0;
    validSubsetIndexes.clear();
    validSubsetIndexes.reserve(std::min((size_t)maxPeaks, subsets.size()));
    for (auto index = 0u; index < subsets.size(); index++)
    {
        auto subsetCounter = subsets[index].first[subsetCounterIndex];
        // Foot keypoints do not affect subsetCounter (too many false positives,
        // same foot usually appears as both left and right keypoints)
        // Pros: Removed tons of false positives
        // Cons: Standalone leg will never be recorded
        if (/*!COCO_CHALLENGE*/false && numberBodyParts == 25)
        {
            // No consider foot keypoints for that
            for (auto i = 19; i < 25; i++)
                subsetCounter -= (subsets[index].first.at(i) > 0);
        }
        const auto subsetScore = subsets[index].second;
        if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) >= minSubsetScore)
        {
            numberPeople++;
            validSubsetIndexes.emplace_back(index);
            if (numberPeople == maxPeaks)
                break;
        }
    }

}

void subsetsToPoseKeypointsAndScores(std::vector<float>& poseKeypoints, std::vector<float>& poseScores, const float scaleFactor,
    const std::vector<std::pair<std::vector<int>, double>>& subsets,
    const std::vector<int>& validSubsetIndexes, const float* const peaksPtr,
    const int numberPeople, const unsigned int numberBodyParts,
    const unsigned int numberBodyPartPairs)
{
    if (numberPeople > 0)
    {
        // Initialized to 0 for non-found keypoints in people
        poseKeypoints.resize(numberPeople*(int)(numberBodyParts) * 3, 0.0f);
        poseScores.resize(numberPeople);
    }
    else
    {
        poseKeypoints.clear();
        poseScores.clear();
    }
    const auto numberBodyPartsAndPAFs = numberBodyParts + numberBodyPartPairs;
    for (auto person = 0u; person < validSubsetIndexes.size(); person++)
    {
        const auto& subsetPair = subsets[validSubsetIndexes[person]];
        const auto& subset = subsetPair.first;
        for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
        {
            const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
            const auto bodyPartIndex = subset[bodyPart];
            if (bodyPartIndex > 0)
            {
                poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
                poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
                poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
            }
        }
        poseScores[person] = subsetPair.second / (float)(numberBodyPartsAndPAFs);
    }

}

void connectBodyPartsCpu(std::vector<float>& poseKeypoints, std::vector<float>& poseScores, const float* const heatMapPtr, const float* const peaksPtr,
    const int & heatMapWidth, const int & heatMapHeight, const int maxPeaks, const int interMinAboveThreshold,
    const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor, std::vector<int>& keypointShape)
{
    // Parts Connection
    keypointShape.resize(3);

    //const std::vector<unsigned int> POSE_COCO_MAP_IDX{ 26,27, 40,41, 48,49, 42,43, 44,45, 50,51, 52,53, 32,33, 28,29, 30,31, 34,35, 36,37, 38,39, 56,57, 58,59, 62,63, 60,61, 64,65, 46,47, 54,55, 66,67,68,69,70,71, 72,73,74,75,76,77 };
    //const auto& mapIdx = POSE_COCO_MAP_IDX;
    const auto& bodyPartPairs = POSE_BODY25B_PAIRS;
    const auto numberBodyParts = 25;

    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

    std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
    const auto subsetCounterIndex = numberBodyParts;
    const auto subsetSize = numberBodyParts;

    const auto peaksOffset = 3 * (maxPeaks + 1);
    const auto heatMapOffset = heatMapWidth*heatMapHeight;


    // std::vector<std::pair<std::vector<int>, double>> refers to:
    //     - std::vector<int>: [body parts locations, #body parts found]
    //     - double: subset score
    const auto subsets = generateInitialSubsets(
        heatMapPtr, peaksPtr, heatMapWidth, heatMapHeight, maxPeaks, interThreshold, interMinAboveThreshold,
        bodyPartPairs, numberBodyParts, numberBodyPartPairs, subsetCounterIndex);

    // Delete people below the following thresholds:
    // a) minSubsetCnt: removed if less than minSubsetCnt body parts
    // b) minSubsetScore: removed if global score smaller than this
    // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above thresholds
    int numberPeople;
    std::vector<int> validSubsetIndexes;
    validSubsetIndexes.reserve(std::min((size_t)maxPeaks, subsets.size()));
    removeSubsetsBelowThresholds(validSubsetIndexes, numberPeople, subsets, subsetCounterIndex,
        numberBodyParts, minSubsetCnt, minSubsetScore, maxPeaks);

    // Fill and return poseKeypoints
    subsetsToPoseKeypointsAndScores(poseKeypoints, poseScores, scaleFactor, subsets, validSubsetIndexes,
        peaksPtr, numberPeople, numberBodyParts, numberBodyPartPairs);

    keypointShape[0] = numberPeople;
    keypointShape[1] = numberBodyParts;
    keypointShape[2] = 3;

}


int poseOpenposeNcnn::Infer(akdData *data,std::vector<inferOption> options,std::vector<inferOutdata> &res){

    ncnn::Net* nettmp = (ncnn::Net*)net;
     const int target_size = 368;
        for (int index = 0; index < data->num; ++index) {

            int img_w = data->widths[index];
            int img_h = data->heights[index];


            int dstw = target_size;
            int dsth = target_size;
            ncnn::Mat in = ncnn::Mat::from_pixels_resize((const unsigned char*)(data->ptr[index]), ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, dstw, dsth);
            in.substract_mean_normalize(mean_vals, scale_vals);

            ncnn::Extractor ex = nettmp->create_extractor();
            ex.set_light_mode(true);
            ex.input("input_input", in);

            std::vector< float > vecResult;

            ncnn::Mat out;
            ex.extract("net_output", out);

            ncnn::Mat outRe;
            ncnn::resize_bilinear(out, outRe, dstw, dsth);

            ncnn::Mat outNms;
            outNms.create(3, POSE_MAX_PEOPLE + 1, out.c);
            nms(outRe, outNms, 0.05f);
            std::vector< float > keypointsdsds;
            std::vector<int> shape;
            std::vector< float > poseScores;
            connectBodyPartsCpu(keypointsdsds, poseScores, (float*)outRe.data, (float*)outNms.data, dstw, dsth, POSE_MAX_PEOPLE, 0.95, 0.05, 3, 0.4, 1, shape);//0,0.0,3,0.4

           inferOutdata vecPerson;
            int kk = 0;
            for (int i = 0; i < shape[0]; i++)
            {
                landmark perPose;
                for (int j = 0; j < shape[1]; j++)
                {
                    KeyPoint kp;
                    kp.x = keypointsdsds[kk++] * img_w / dstw;
                    kp.y = keypointsdsds[kk++] * img_h / dsth;
                    kp.prob=keypointsdsds[kk++];
                    perPose.pp.push_back(kp);
                    //perPose.pp_probs.push_back(keypointsdsds[kk++]);
                }
                vecPerson.lms.push_back(perPose);
            }
            res .push_back(vecPerson);

        }

//    ncnn::Net* yolov5=(ncnn::Net*)net;
//    const int target_size = 368;
//    for(int index=0; index<data->num; ++index)
//    {
//        int img_w = data->widths[index];
//        int img_h = data->heights[index];

//        int w = target_size;
//        int h = target_size;
//        //  float scale = 1.f;
//        //        if (w > h)
//        //        {
//        //            scale = (float)target_size / w;
//        //            w = target_size;
//        //            h = h * scale;
//        //        }
//        //        else
//        //        {
//        //            scale = (float)target_size / h;
//        //            h = target_size;
//        //            w = w * scale;
//        //        }

//        ncnn::Mat in = ncnn::Mat::from_pixels_resize((const unsigned char*)(data->ptr[index]), ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

//        //        int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
//        //        int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
//        //        ncnn::Mat in_pad;
//        //        ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

//        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
//        in.substract_mean_normalize(0, norm_vals);

//        ncnn::Extractor ex = yolov5->create_extractor();

//        ex.input("input", in);
//        ncnn::Mat out;
//        ex.extract("net_output", out);
//        inferOutdata outdatatmp;
//        genoutput(out,outdatatmp);
//        res.push_back(outdatatmp);

//    }



}
int poseOpenposeNcnn::GetPairs(std::vector<int> &pairs)
{
    pairs.clear();
    for(int i=0;i<POSE_BODY25B_PAIRS.size();++i)
    {
        pairs.push_back(POSE_BODY25B_PAIRS[i]);

    }
//    return POSE_BODY25B_PAIRS;
    return 0;
}
int poseOpenposeNcnn::preprocess_cpu(akdData *data,int inferw,int inferh){

}
int poseOpenposeNcnn::postprocess_cpu(std::vector<std::vector<int>> outputsizes,std::vector<inferOutdata> &res){

}
lmService* poseinit_openpose_ncnn(lminitconfig config)
{
    poseOpenposeNcnn * ch=new poseOpenposeNcnn(config.modelpath,config.gpu,config.threadnum,config.maxnetsize,config.maxbatchsize);
    if(ch->initsuccess<0)
    {
        return nullptr;
    }
    return ch;
}

#include"pose_openpose_trt.h"
#include"3rdwrap/trt/trtService.h"
#include"3rdwrap/cuda/cudaService.h"
#include"opencv2/opencv.hpp"
const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
    {0,  "Nose"},
    {1,  "Neck"},
    {2,  "RShoulder"},
    {3,  "RElbow"},
    {4,  "RWrist"},
    {5,  "LShoulder"},
    {6,  "LElbow"},
    {7,  "LWrist"},
    {8,  "MidHip"},
    {9,  "RHip"},
    {10, "RKnee"},
    {11, "RAnkle"},
    {12, "LHip"},
    {13, "LKnee"},
    {14, "LAnkle"},
    {15, "REye"},
    {16, "LEye"},
    {17, "REar"},
    {18, "LEar"},
    {19, "LBigToe"},
    {20, "LSmallToe"},
    {21, "LHeel"},
    {22, "RBigToe"},
    {23, "RSmallToe"},
    {24, "RHeel"},
    {25, "Background"}
};
// BODY_25
const   std::vector<int> POSE_BODY25_PAIRS={0,15,0,16,15,17,16,18,0,1,1,2,2,3,3,4,1,5,5,6,6,7,1,8,
                                            8,9,9,10,10,11,11,24,11,22,22,23,8,12,12,13,13,14,14,21,21,19,19,20};
// BODY_25
std::vector<unsigned int>POSE_BODY25_MAP_IDX {
    0,1, 14,15, 22,23, 16,17, 18,19, 24,25, 26,27, 6,7, 2,3, 4,5, 8,9, 10,11, 12,13, 30,31, 32,33, 36,37, 34,35, 38,39, 20,21, 28,29, 40,41,42,43,44,45, 46,47,48,49,50,51
};

// coco
const std::vector<unsigned int> POSE_COCO_PAIRS{ 1, 2, 1, 5, 2, 3, 3, 4, 5, 6, 6, 7, 1, 8, 8, 9, 9, 10, 1, 11, 11, 12, 12, 13, 1, 0, 0, 14, 14, 16, 0, 15, 15, 17, 2, 16, 5, 17 };
const std::vector<unsigned int> POSE_COCO_MAP_IDX{ 31, 32, 39, 40, 33, 34, 35, 36, 41, 42, 43, 44, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 47, 48, 49, 50, 53, 54, 51, 52, 55, 56, 37, 38, 45, 46 };
const unsigned int POSE_MAX_PEOPLE = 96;

#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <functional>
template<typename T>
inline int intRound(const T a)
{
    return int(a+0.5f);
}
// Max/min functions
template<typename T>
inline T fastMax(const T a, const T b)
{
    return (a > b ? a : b);
}

template<typename T>
inline T fastMin(const T a, const T b)
{
    return (a < b ? a : b);
}

poseOpenposeTrt::poseOpenposeTrt(std::string modelpath,int gpu,int threadnum,int modelsize,int maxbatchsize)
{
    trtConfig config;
    config.gpuindex=gpu;
    config.modelpath=modelpath;
    config.maxbatchsize=maxbatchsize;
    config.inputmaxsize=modelsize;
    config.inputminsize=modelsize;
    //config.mt=ModelType::caffe;
    trtIOprofile ioparam;
    trtService* trtnew=new trtService (config,ioparam);
    maxinputsize=ioparam.maxinputsize[0];
    maxoutputsize=ioparam.maxoutputsize;

    int nn=1+maxoutputsize.size();
    gmm.clear();
    gmm.resize(nn);
    cmm.clear();
    cmm.resize(nn);

    long maxinputsizeall=maxinputsize[0]*maxinputsize[1]*maxinputsize[2]*maxinputsize[3];
    gmm[0]=akdcuda::safeCudaMalloc(maxinputsizeall*sizeof(float));
    cmm[0]=new float[maxinputsizeall];

    for(int i=1;i<nn;++i){
        long maxoutputsizeall1=1;
        for(int j=0;j<maxoutputsize[i-1].size();++j)
        {
            maxoutputsizeall1*=maxoutputsize[i-1][j];
        }
        gmm[i]=akdcuda::safeCudaMalloc(maxoutputsizeall1*sizeof(float));
        cmm[i]=new float[maxoutputsizeall1];

    }
    //     std::cout<<"init cmm"<<std::endl;
    //    for(int i=0;i<10;++i)
    //    {
    //        std::cout<<*(cmm[1]+i)<<std::endl;
    //    }
    trt=trtnew;
    initsuccess=1;
}
poseOpenposeTrt::~poseOpenposeTrt()
{
    if(trt!=nullptr)
    {
        delete (trtService*)trt;
        trt=nullptr;
    }
    for(int i=0;i<gmm.size();++i)
    {
        cudaFree(gmm[i]);
        gmm[i]=nullptr;
        delete cmm[i];
        cmm[i]=nullptr;
    }


}
int poseOpenposeTrt::preprocess_cpu(akdData *data,int inferw,int inferh)
{
    for(int i=0;i<data->num;++i)
    {
        float* d_in=(float*)(gmm[0])+inferw*inferh*3*i;
        cv::Mat img(data->heights[i],data->widths[i],CV_8UC3,data->ptr[i]);
        cv::Mat img_tmp;// = imgs[i].clone();
        img.convertTo(img_tmp, CV_32F, 1 / 256.f, -0.5);
        cv::resize(img_tmp,img_tmp,cv::Size(inferw,inferh));
        //img_tmp=img_tmp/255.0;
        std::vector<cv::Mat> channels;
        cv::split(img_tmp, channels);
        int inputcount=inferh*inferw;

        cudaMemcpy((void*)d_in, channels[2].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount), channels[1].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount*2), channels[0].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);

    }
    return 0;
}

void connectBodyPartsCpu(std::vector<float>& poseKeypoints, const float* const heatMapPtr, const float* const peaksPtr,
                         const cv::Size& heatMapSize, const int maxPeaks, const int interMinAboveThreshold,
                         const float interThreshold, const int minSubsetCnt, const float minSubsetScore, const float scaleFactor,
                         std::vector<int>& keypointShape)
{
    keypointShape.resize(3);

    const auto& bodyPartPairs = POSE_BODY25_PAIRS;
    std::vector<unsigned int> mapIdx = POSE_BODY25_MAP_IDX;
    for(int i=0;i<mapIdx.size();++i)
    {
        mapIdx[i]=mapIdx[i]+26;
    }
    const auto numberBodyParts = 25;

    const auto numberBodyPartPairs = bodyPartPairs.size() / 2;

    std::vector<std::pair<std::vector<int>, double>> subset;    // Vector<int> = Each body part + body parts counter; double = subsetScore
    const auto subsetCounterIndex = numberBodyParts;
    const auto subsetSize = numberBodyParts + 1;

    const auto peaksOffset = 3 * (maxPeaks + 1);
    const auto heatMapOffset = heatMapSize.area();

    for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; pairIndex++)
    {
        const auto bodyPartA = bodyPartPairs[2 * pairIndex];
        const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
        const auto* candidateA = peaksPtr + bodyPartA*peaksOffset;
        const auto* candidateB = peaksPtr + bodyPartB*peaksOffset;
        const auto nA = intRound(candidateA[0]);
        const auto nB = intRound(candidateB[0]);

        // add parts into the subset in special case
        if (nA == 0 || nB == 0)
        {
            // Change w.r.t. other
            if (nA == 0) // nB == 0 or not
            {
                for (auto i = 1; i <= nB; i++)
                {
                    bool num = false;
                    const auto indexB = bodyPartB;
                    for (auto j = 0u; j < subset.size(); j++)
                    {
                        const auto off = (int)bodyPartB*peaksOffset + i * 3 + 2;
                        if (subset[j].first[indexB] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        rowVector[bodyPartB] = bodyPartB*peaksOffset + i * 3 + 2; //store the index
                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
                        const auto subsetScore = candidateB[i * 3 + 2]; //second last number in each row is the total score
                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
            else // if (nA != 0 && nB == 0)
            {
                for (auto i = 1; i <= nA; i++)
                {
                    bool num = false;
                    const auto indexA = bodyPartA;
                    for (auto j = 0u; j < subset.size(); j++)
                    {
                        const auto off = (int)bodyPartA*peaksOffset + i * 3 + 2;
                        if (subset[j].first[indexA] == off)
                        {
                            num = true;
                            break;
                        }
                    }
                    if (!num)
                    {
                        std::vector<int> rowVector(subsetSize, 0);
                        rowVector[bodyPartA] = bodyPartA*peaksOffset + i * 3 + 2; //store the index
                        rowVector[subsetCounterIndex] = 1; //last number in each row is the parts number of that person
                        const auto subsetScore = candidateA[i * 3 + 2]; //second last number in each row is the total score
                        subset.emplace_back(std::make_pair(rowVector, subsetScore));
                    }
                }
            }
        }
        else // if (nA != 0 && nB != 0)
        {
            std::vector<std::tuple<double, int, int>> temp;
            const auto numInter = 10;
            const auto* const mapX = heatMapPtr + mapIdx[2 * pairIndex] * heatMapOffset;
            const auto* const mapY = heatMapPtr + mapIdx[2 * pairIndex + 1] * heatMapOffset;
            for (auto i = 1; i <= nA; i++)
            {
                for (auto j = 1; j <= nB; j++)
                {
                    const auto dX = candidateB[j * 3] - candidateA[i * 3];
                    const auto dY = candidateB[j * 3 + 1] - candidateA[i * 3 + 1];
                    const auto normVec = float(std::sqrt(dX*dX + dY*dY));
                    // If the peaksPtr are coincident. Don't connect them.
                    if (normVec > 1e-6)
                    {
                        const auto sX = candidateA[i * 3];
                        const auto sY = candidateA[i * 3 + 1];
                        const auto vecX = dX / normVec;
                        const auto vecY = dY / normVec;

                        auto sum = 0.;
                        auto count = 0;
                        for (auto lm = 0; lm < numInter; lm++)
                        {
                            const auto mX = fastMin(heatMapSize.width - 1, intRound(sX + lm*dX / numInter));
                            const auto mY = fastMin(heatMapSize.height - 1, intRound(sY + lm*dY / numInter));
                            //checkGE(mX, 0, "", __LINE__, __FUNCTION__, __FILE__);
                            //checkGE(mY, 0, "", __LINE__, __FUNCTION__, __FILE__);
                            const auto idx = mY * heatMapSize.width + mX;
                            const auto score = (vecX*mapX[idx] + vecY*mapY[idx]);
                            if (score > interThreshold)
                            {
                                sum += score;
                                count++;
                            }
                        }

                        // parts score + connection score
                        if (count > interMinAboveThreshold)
                            temp.emplace_back(std::make_tuple(sum / count, i, j));
                    }
                }
            }

            // select the top minAB connection, assuming that each part occur only once
            // sort rows in descending order based on parts + connection score
            if (!temp.empty())
                std::sort(temp.begin(), temp.end(), std::greater<std::tuple<float, int, int>>());

            std::vector<std::tuple<int, int, double>> connectionK;

            const auto minAB = fastMin(nA, nB);
            std::vector<int> occurA(nA, 0);
            std::vector<int> occurB(nB, 0);
            auto counter = 0;
            for (auto row = 0u; row < temp.size(); row++)
            {
                const auto score = std::get<0>(temp[row]);
                const auto x = std::get<1>(temp[row]);
                const auto y = std::get<2>(temp[row]);
                if (!occurA[x - 1] && !occurB[y - 1])
                {
                    connectionK.emplace_back(std::make_tuple(bodyPartA*peaksOffset + x * 3 + 2,
                                                             bodyPartB*peaksOffset + y * 3 + 2,
                                                             score));
                    counter++;
                    if (counter == minAB)
                        break;
                    occurA[x - 1] = 1;
                    occurB[y - 1] = 1;
                }
            }

            // Cluster all the body part candidates into subset based on the part connection
            // initialize first body part connection 15&16
            if (pairIndex == 0)
            {
                for (const auto connectionKI : connectionK)
                {
                    std::vector<int> rowVector(numberBodyParts + 3, 0);
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    const auto score = std::get<2>(connectionKI);
                    rowVector[bodyPartPairs[0]] = indexA;
                    rowVector[bodyPartPairs[1]] = indexB;
                    rowVector[subsetCounterIndex] = 2;
                    // add the score of parts and the connection
                    const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                    subset.emplace_back(std::make_pair(rowVector, subsetScore));
                }
            }
            // Add ears connections (in case person is looking to opposite direction to camera)
            else if (pairIndex == 17 || pairIndex == 18)
            {
                for (const auto& connectionKI : connectionK)
                {
                    const auto indexA = std::get<0>(connectionKI);
                    const auto indexB = std::get<1>(connectionKI);
                    for (auto& subsetJ : subset)
                    {
                        auto& subsetJFirst = subsetJ.first[bodyPartA];
                        auto& subsetJFirstPlus1 = subsetJ.first[bodyPartB];
                        if (subsetJFirst == indexA && subsetJFirstPlus1 == 0)
                            subsetJFirstPlus1 = indexB;
                        else if (subsetJFirstPlus1 == indexB && subsetJFirst == 0)
                            subsetJFirst = indexA;
                    }
                }
            }
            else
            {
                if (!connectionK.empty())
                {
                    // A is already in the subset, find its connection B
                    for (auto i = 0u; i < connectionK.size(); i++)
                    {
                        const auto indexA = std::get<0>(connectionK[i]);
                        const auto indexB = std::get<1>(connectionK[i]);
                        const auto score = std::get<2>(connectionK[i]);
                        auto num = 0;
                        for (auto j = 0u; j < subset.size(); j++)
                        {
                            if (subset[j].first[bodyPartA] == indexA)
                            {
                                subset[j].first[bodyPartB] = indexB;
                                num++;
                                subset[j].first[subsetCounterIndex] = subset[j].first[subsetCounterIndex] + 1;
                                subset[j].second = subset[j].second + peaksPtr[indexB] + score;
                            }
                        }
                        // if can not find partA in the subset, create a new subset
                        if (num == 0)
                        {
                            std::vector<int> rowVector(subsetSize, 0);
                            rowVector[bodyPartA] = indexA;
                            rowVector[bodyPartB] = indexB;
                            rowVector[subsetCounterIndex] = 2;
                            const auto subsetScore = peaksPtr[indexA] + peaksPtr[indexB] + score;
                            subset.emplace_back(std::make_pair(rowVector, subsetScore));
                        }
                    }
                }
            }
        }
    }

    // Delete people below the following thresholds:
    // a) minSubsetCnt: removed if less than minSubsetCnt body parts
    // b) minSubsetScore: removed if global score smaller than this
    // c) POSE_MAX_PEOPLE: keep first POSE_MAX_PEOPLE people above thresholds
    auto numberPeople = 0;
    std::vector<int> validSubsetIndexes;
    validSubsetIndexes.reserve(fastMin((size_t)POSE_MAX_PEOPLE, subset.size()));
    for (auto index = 0u; index < subset.size(); index++)
    {
        const auto subsetCounter = subset[index].first[subsetCounterIndex];
        const auto subsetScore = subset[index].second;
        if (subsetCounter >= minSubsetCnt && (subsetScore / subsetCounter) > minSubsetScore)
        {
            numberPeople++;
            validSubsetIndexes.emplace_back(index);
            if (numberPeople == POSE_MAX_PEOPLE)
                break;
        }
        else if (subsetCounter < 1)
            printf("Bad subsetCounter. Bug in this function if this happens. %d, %s, %s", __LINE__, __FUNCTION__, __FILE__);
    }

    // Fill and return poseKeypoints
    keypointShape = { numberPeople, (int)numberBodyParts, 3 };
    if (numberPeople > 0)
        poseKeypoints.resize(numberPeople * (int)numberBodyParts * 3);
    else
        poseKeypoints.clear();

    for (auto person = 0u; person < validSubsetIndexes.size(); person++)
    {
        const auto& subsetI = subset[validSubsetIndexes[person]].first;
        for (auto bodyPart = 0u; bodyPart < numberBodyParts; bodyPart++)
        {
            const auto baseOffset = (person*numberBodyParts + bodyPart) * 3;
            const auto bodyPartIndex = subsetI[bodyPart];
            if (bodyPartIndex > 0)
            {
                poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
                poseKeypoints[baseOffset + 1] = peaksPtr[bodyPartIndex - 1] * scaleFactor;
                poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
            }
            else
            {
                poseKeypoints[baseOffset] = 0.f;
                poseKeypoints[baseOffset + 1] = 0.f;
                poseKeypoints[baseOffset + 2] = 0.f;
            }
        }
    }
}

//topShape[1] = bottomShape[1] - 1; // Number parts + bck - 1      56 = 57 - 1
//topShape[2] = maxPeaks + 1; // # maxPeaks + 1                    97 = 96 + 1
//topShape[3] = 3;  // X, Y, score                                 3


void nms(float* bottom_blob, float* top_blob, float threshold,int w,int h,int toph,int topw,int num,int mapnum){

    //int w = bottom_blob->width;
    //int h = bottom_blob->height;
    int plane_offset = w * h;
    float* ptr = bottom_blob;
    float* top_ptr = top_blob;
    int top_plane_offset = topw * toph;
    int max_peaks = toph - 1;

    for (int n = 0; n < num; ++n){
        for (int c = 0; c < mapnum-1; ++c){

            int num_peaks = 0;
            for (int y = 1; y < h - 1 && num_peaks != max_peaks; ++y){
                for (int x = 1; x < w - 1 && num_peaks != max_peaks; ++x){
                    float value = ptr[y*w + x];
                    if (value > threshold){
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

                            float xAcc = 0;
                            float yAcc = 0;
                            float scoreAcc = 0;
                            for (int kx = -3; kx <= 3; ++kx){
                                int ux = x + kx;
                                if (ux >= 0 && ux < w){
                                    for (int ky = -3; ky <= 3; ++ky){
                                        int uy = y + ky;
                                        if (uy >= 0 && uy < h){
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
           // std::cout<<"channel:"<<c<<" peaks: "<<num_peaks<<std::endl;
            ptr += plane_offset;
            top_ptr += top_plane_offset;
        }
    }
}

int GetPeaks(akdData *data,std::vector<std::vector<int>> &outputsizes,void*gmm,float*cmm,std::vector<float> &peaks)
{
    long outputsize=outputsizes[0][0]*outputsizes[0][1]*outputsizes[0][2]*outputsizes[0][3];
    cudaMemcpy(cmm,gmm,outputsize*sizeof (float),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    //std::cout<<"out cmm"<<std::endl;
    int num=outputsizes[0][0];
    int mapnum=outputsizes[0][1];
    int w=outputsizes[0][2];
    int h=outputsizes[0][3];
    int outputsize1=mapnum*w*h;
    int featmapsize=w*h;
    int pointfeatnum=26;
    int pairfeatnum=26;
    float scalew=data->widths[0]/w;
    float scaleh=data->heights[0]/h;
    float threshold=0.3;
    for(int i=0;i<num;++i)
    {
        inferOutdata poses;

        float* feats=cmm+outputsize1*i;
        landmark lm;
        for(int j=0;j<25;++j)
        {
            float* ptr=feats+j*featmapsize;
            for(int y=1;y<h-1;++y)
            {
                for(int x=1;x<w-1;++x)
                {
                    float value = ptr[y*w + x];
                    if (value <= threshold){
                        continue;
                    }
                    const float topLeft = ptr[(y - 1)*w + x - 1];
                    const float top = ptr[(y - 1)*w + x];
                    const float topRight = ptr[(y - 1)*w + x + 1];
                    const float left = ptr[y*w + x - 1];
                    const float right = ptr[y*w + x + 1];
                    const float bottomLeft = ptr[(y + 1)*w + x - 1];
                    const float bottom = ptr[(y + 1)*w + x];
                    const float bottomRight = ptr[(y + 1)*w + x + 1];

                    if (value <= topLeft || value <= top || value <= topRight
                            ||  value <= left ||  value <= right
                            ||  value <= bottomLeft ||  value <= bottom ||  value <= bottomRight)
                    {
                        continue;
                    }
                    //计算亚像素坐标
                    float xAcc = 0;
                    float yAcc = 0;
                    float scoreAcc = 0;
                    for (int kx = -3; kx <= 3; ++kx){
                        int ux = x + kx;
                        if (ux >= 0 && ux < w){
                            for (int ky = -3; ky <= 3; ++ky){
                                int uy = y + ky;
                                if (uy >= 0 && uy < h){
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

                    yAcc*=scaleh;
                    xAcc*=scalew;
                    lm.pp.push_back(cvpoint(xAcc,yAcc));
                    lm.pp_probs.push_back(scoreAcc);

                    //                    peaks.push_back(xAcc);
                    //                    peaks.push_back(yAcc);
                    //                    peaks.push_back(scoreAcc);

                }
            }

        }
        // poses.lms.push_back(lm);
        //res.push_back(poses);
    }
    return 0;
}
int poseOpenposeTrt::postprocess_cpu(akdData *data,std::vector<std::vector<int>> &outputsizes,std::vector<inferOutdata> &res)
{
    //post process cpu
    long outputsize=outputsizes[0][0]*outputsizes[0][1]*outputsizes[0][2]*outputsizes[0][3];
    cudaMemcpy(cmm[1],gmm[1],outputsize*sizeof (float),cudaMemcpyKind::cudaMemcpyDeviceToHost);

    //std::cout<<"out cmm"<<std::endl;
    int num=outputsizes[0][0];
    int mapnum=outputsizes[0][1];
    int w=outputsizes[0][2];
    int h=outputsizes[0][3];
    int outputsize1=mapnum*w*h;///78*46*46
    int featmapsize=w*h;
    int pointfeatnum=25;
    int pairfeatnum=25;
    float scalew=data->widths[0]/w;
    float scaleh=data->heights[0]/h;
    float threshold=0.3;
    for(int i=0;i<num;++i)
    {
        inferOutdata poses;

        float* feats=cmm[1]+outputsize1*i;
        landmark lm;
        lm.pp.resize(25);
        lm.pp_probs.resize(25,-1);
        for(int j=0;j<25;++j)
        {
            float* ptr=feats+j*featmapsize;
            for(int y=1;y<h-1;++y)
            {
                for(int x=1;x<w-1;++x)
                {
                    float value = ptr[y*w + x];
                    if (value <= threshold){
                        continue;
                    }
                    const float topLeft = ptr[(y - 1)*w + x - 1];
                    const float top = ptr[(y - 1)*w + x];
                    const float topRight = ptr[(y - 1)*w + x + 1];
                    const float left = ptr[y*w + x - 1];
                    const float right = ptr[y*w + x + 1];
                    const float bottomLeft = ptr[(y + 1)*w + x - 1];
                    const float bottom = ptr[(y + 1)*w + x];
                    const float bottomRight = ptr[(y + 1)*w + x + 1];

                    if (value <= topLeft || value <= top || value <= topRight
                            ||  value <= left ||  value <= right
                            ||  value <= bottomLeft ||  value <= bottom ||  value <= bottomRight)
                    {
                        continue;
                    }
                    //计算亚像素坐标
                    float xAcc = 0;
                    float yAcc = 0;
                    float scoreAcc = 0;
                    for (int kx = -3; kx <= 3; ++kx){
                        int ux = x + kx;
                        if (ux >= 0 && ux < w){
                            for (int ky = -3; ky <= 3; ++ky){
                                int uy = y + ky;
                                if (uy >= 0 && uy < h){
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

                    yAcc*=scaleh;
                    xAcc*=scalew;
                    if(scoreAcc>lm.pp_probs[j]){
                        lm.pp[j]=cvpoint(xAcc,yAcc);
                        lm.pp_probs[j]=scoreAcc;}
                }
            }
        }

        poses.lms.push_back(lm);
        res.push_back(poses);
    }
    return 0;
}

int poseOpenposeTrt::Infer(akdData *data,std::vector<inferOption> options,std::vector<inferOutdata> &res)
{
    trtService* runner=(trtService*) trt;
    // preprocess cpu
    int picnum=data->num;
    int inferw=options[0].inferw;
    int inferh=options[0].inferh;
    double t1=cv::getTickCount();
    preprocess_cpu(data,inferw,inferh);
    double t2=cv::getTickCount();
    std::vector<std::vector<int>> outputsizes;

    runner->infer(gmm.data(),picnum,inferw,inferh,outputsizes);

    double t3=cv::getTickCount();

    long outputsize=outputsizes[0][0]*outputsizes[0][1]*outputsizes[0][2]*outputsizes[0][3];
    cudaMemcpy(cmm[1],gmm[1],outputsize*sizeof(float),cudaMemcpyKind::cudaMemcpyDeviceToHost);

    cv::Size baseSize = cv::Size(inferw, inferh);  //Size(368, 368);
    //    float* nms_out = createBlob_local(1, 56, 99 + 1, 3);
    int mapnum=picnum*78;//outputsizes[0][0]*outputsizes[0][1];
    float* nms_out=new float[picnum*77*(POSE_MAX_PEOPLE+1)*3];
    float* input = new float[baseSize.height*baseSize.width*mapnum];//createBlob_local(1, 57, baseSize.height, baseSize.width);
    for (int i = 0; i < mapnum; ++i){
        cv::Mat um(baseSize.height, baseSize.width, CV_32F, input + baseSize.height*baseSize.width*i);
        resize(cv::Mat(46, 46, CV_32F, cmm[1] + 46*46*i), um, baseSize, 0, 0, cv::INTER_CUBIC);
    }

    nms(input, nms_out, 0.05,baseSize.width,baseSize.height,(POSE_MAX_PEOPLE+1),3,picnum,78);
    float scalew=(float)(data->widths[0]*1.0)/((float)inferw);
    float scaleh=(float)(data->heights[0]*1.0)/((float)inferh);
    for(int i=0;i<picnum;++i){
        std::vector<float> keypoints;
        std::vector<int> shape;

        connectBodyPartsCpu(keypoints, input+368*368*78*i, nms_out+(POSE_MAX_PEOPLE+1)*3*77*i, baseSize, POSE_MAX_PEOPLE, 5, 0.05, 3, 0.1, 1, shape);
        inferOutdata da;
        for(int j=0;j<shape[0];++j)
        {
            landmark lm;
            for(int jj=0;jj<shape[1];++jj)
            {
                //int
                lm.pp.push_back(cvpoint(keypoints[jj*3]*scalew,keypoints[jj*3+1]*scaleh));
                lm.pp_probs.push_back(keypoints[jj*3+2]);
            }
            da.lms.push_back(lm);
        }
        res.push_back(da);
    }
    delete []nms_out;
    delete []input;
    //postprocess_cpu(data,outputsizes,res);
    double t4=cv::getTickCount();
    std::cout<<picnum<<"time:"<<(t2-t1)*1000/cv::getTickFrequency()<<" ";
    std::cout<<(t3-t2)*1000/cv::getTickFrequency()<<" ";
    std::cout<<(t4-t3)*1000/cv::getTickFrequency()<<std::endl;
    return -1;
}

int poseOpenposeTrt::GetPairs(std::vector<int> &pairs)
{

    // BODY_25
    std::vector<int> pp={0,15,0,16,15,17,16,18,0,1,1,2,2,3,3,4,1,5,5,6,6,7,1,8,
                         8,9,9,10,10,11,11,24,11,22,22,23,8,12,12,13,13,14,14,21,21,19,19,20};
    // coco 18

    pairs=pp;
    return 0;
}

lmService* poseinit_openpose_trt(lminitconfig config)
{
    poseOpenposeTrt * ch=new poseOpenposeTrt(config.modelpath,config.gpu,config.threadnum,config.maxnetsize,config.maxbatchsize);
    if(ch->initsuccess<0)
    {
        return nullptr;
    }
    return ch;
}

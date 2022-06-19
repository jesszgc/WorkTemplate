#ifndef POSE_OPENPOSE_NCNN_H
#define POSE_OPENPOSE_NCNN_H
#include "../lmService.h"
#include "../../AkdComm.h"
#include<string>
#include<vector>
class poseOpenposeNcnn : public lmService
{
public:
    poseOpenposeNcnn(std::string modelpath,int gpu,int threadnum,int modelsize=640,int maxbatchsize=1);
    ~poseOpenposeNcnn();
    int Infer(akdData *data,std::vector<inferOption> options,std::vector<inferOutdata> &res);
    int GetPairs(std::vector<int> &pairs);
private:
    int preprocess_cpu(akdData *data,int inferw,int inferh);
    int postprocess_cpu(std::vector<std::vector<int>> outputsizes,std::vector<inferOutdata> &res);
public:
    int initsuccess=-1;
private:
    void *net=nullptr;
    //const float mean_vals[3] = { 0.485f * 255.f, 0.456f * 255.f, 0.406f * 255.f };
    //const float norm_vals[3] = { 1 / 0.229f / 255.f, 1 / 0.224f / 255.f, 1 / 0.225f / 255.f };

    float mean_vals[3]={ 127.5f, 127.5f, 127.5f };
    float scale_vals[3]= { 0.0078125f, 0.0078125f, 0.0078125f };

    int netsize=640;
};

#endif

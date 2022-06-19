#ifndef POSE_OPENPOSE_TRT_H
#define POSE_OPENPOSE_TRT_H
#include "../lmService.h"
#include "../../AkdComm.h"
#include<string>
#include<vector>
class poseOpenposeTrt : public lmService
{
public:
    poseOpenposeTrt(std::string modelpath,int gpu,int threadnum,int modelsize=640,int maxbatchsize=1);
    ~poseOpenposeTrt();
    int Infer(akdData *data,std::vector<inferOption> options,std::vector<inferOutdata> &res);
    int GetPairs(std::vector<int> &pairs);
private:
    int preprocess_cpu(akdData *data,int inferw,int inferh);
    int postprocess_cpu(akdData *data,std::vector<std::vector<int>> &outputsizes,std::vector<inferOutdata> &res);
public:
    int initsuccess=-1;
private:
    void *trt=nullptr;
    std::vector<int> maxinputsize;
    std::vector<int> mininputsize;
    std::vector<std::vector<int>> maxoutputsize;
    int netsize=640;
    std::vector<void*> gmm{nullptr};
    std::vector<float*> cmm{nullptr};
};

#endif

#ifndef POSE_LWOPENPOSE_TRT_H
#define POSE_LWOPENPOSE_TRT_H
#include "../lmService.h"
#include "../../AkdComm.h"
#include<string>
#include<vector>
class poseLwOpenposeTrt : public lmService
{
public:
    poseLwOpenposeTrt(std::string modelpath,int gpu,int threadnum,int modelsize=640,int maxbatchsize=1);
    ~poseLwOpenposeTrt();
    int Infer(akdData *data,std::vector<inferOutdata> &res);
private:
    int preprocess_cpu(akdData *data);
    int postprocess_cpu(std::vector<std::vector<int>> outputsizes,std::vector<inferOutdata> &res);
public:
    int initsuccess=-1;
private:
    void *trt=nullptr;
    int netsize=640;
    std::vector<void*> gmm{nullptr};
    std::vector<float*> cmm{nullptr};
};

#endif

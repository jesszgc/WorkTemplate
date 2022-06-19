#ifndef FRARCFACETRT_H
#define FRARCFACETRT_H
#include"AkdComm.h"
#include"../frService.h"

class frArcfaceTrt:public frService
{
    public:
     frArcfaceTrt(std::string modelpath,int gpu,int threadnum,int maxbatchsize=1);
     ~frArcfaceTrt();
     int Extract(akdData *data,std::vector<std::vector<float>> &res);
private:
     int preprocess_cpu(akdData*data,int w,int h);
     int postprocess_cpu(std::vector<std::vector<int>> outputsizes,std::vector<std::vector<float>> &res);
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
    float *input=nullptr;
    float* nms_out=nullptr;

    int inferw=-1;
    int inferh=-1;
    
};




#endif

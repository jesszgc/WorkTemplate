#ifndef LMSERVICE_H
#define LMSERVICE_H
#include "AkdComm.h"

struct KeyPoint{
    int x=0;
    int y=0;
    float prob=-1;
};
struct landmark{
    std::vector<KeyPoint> pp;
    cvrect bbox;
//    std::vector<cvpoint> pp;
//    std::vector<float>pp_probs;
};
struct inferOutdata{
     std::vector<cvrect> locs;
     std::vector<landmark> lms;
     int wholeimage=-1;
};
struct inferOption
{
    std::vector<cvrect> locs;
    int wholeimage=-1;
    int inferw=368;
    int inferh=368;
};
struct lminitconfig{
    std::string modelpath;
    int gpu;
    int threadnum;
    int maxbatchsize;
    int maxnetsize;
};

class lmService
{
    public:
     lmService();
      virtual ~lmService()=0;
    
   int virtual Infer(akdData *data,std::vector<inferOption> options,std::vector<inferOutdata> &res)=0;
   int virtual GetPairs(std::vector<int> &pairs)=0;
    
};


EXPORT lmService* poseinit_openpose_trt(lminitconfig config);
EXPORT lmService* poseinit_openpose_ncnn(lminitconfig config);
EXPORT int lmServiceRelease(lmService* od);

#endif

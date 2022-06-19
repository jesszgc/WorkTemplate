#ifndef FRSERVICE_H
#define FRSERVICE_H
#include"AkdComm.h"

struct frinitconfig{
    std::string modelpath;
    int gpu;
    int threadnum;
    int maxbatchsize;
    int maxnetsize;
};
class frService
{
    public:
     frService();
      virtual ~frService()=0;
    
   int virtual Extract(akdData *data,std::vector<std::vector<float>> &res)=0;
    
};


EXPORT frService* frinit_arcface_ncnn(frinitconfig config);
EXPORT frService* frinit_arcface_trt(frinitconfig config);
EXPORT int frServiceRelease(frService* od);

#endif

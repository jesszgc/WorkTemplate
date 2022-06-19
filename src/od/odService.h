#ifndef ODSERVICE_H
#define ODSERVICE_H
#include"AkdComm.h"

struct objBox{

    cvrect rect;
    int label;
    std::string labelname;
    float prob;
    std::vector<cvpoint> pp;
    
};

struct odinitconfig{
    std::string modelpath;
    int gpu;
    int threadnum;
    int maxbatchsize;
    int maxnetsize;
};
class odService
{
    public:
     odService();
      virtual ~odService()=0;
    
   int virtual Detect(akdData *data,std::vector<std::vector<objBox>> &res)=0;
    
};


EXPORT odService* bodydetectinit_yolov5_ncnn(odinitconfig config);
EXPORT odService* bodydetectinit_yolov5_trt(odinitconfig config);
EXPORT odService* bodydetectinit_yolov5_3o_trt(odinitconfig config);

EXPORT odService* headdetectinit_yolov5_trt(odinitconfig config);
EXPORT odService* facedetectinit_retinaface_ncnn(odinitconfig config);

EXPORT int odServiceRelease(odService* od);

#endif

#ifndef HEADDETECT_YOLOV5_TRT_H
#define HEADDETECT_YOLOV5_TRT_H

#include "../odService.h"

class headDetectYolov5Trt : public odService
{
public:
    headDetectYolov5Trt(std::string modelpath,int gpu,int threadnum,int modelsize=640,int maxbatchsize=1);
    ~headDetectYolov5Trt();
    int Detect(akdData *data,std::vector<std::vector<objBox>> &res);
public:
    int initsuccess=-1;
private:
    void *net=nullptr;
    int netsize=640;
};

#endif // BODYDETECT_YOLOV5_TRT_H

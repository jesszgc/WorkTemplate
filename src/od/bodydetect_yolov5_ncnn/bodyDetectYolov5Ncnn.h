
#ifndef BODYDETECTYOLOV5NCNN_H
#define BODYDETECTYOLOV5NCNN_H
#include"../odService.h"

class bodyDetectYolov5Ncnn:public odService
{
public:
    bodyDetectYolov5Ncnn(std::string modelpath,int gpu,int threadnum,int modelsize=640);
    ~bodyDetectYolov5Ncnn();
    int Detect(akdData *data,std::vector<std::vector<objBox>> &res);
public:
    int initsuccess=-1;
private:
    void *net=nullptr;
    int netsize;
    
};
#endif

#ifndef FACEDETECT_RETINAFACE_NCNN_H
#define FACEDETECT_RETINAFACE_NCNN_H

#include "../odService.h"

class faceDetectretinaNcnn : public odService
{
public:
    faceDetectretinaNcnn(std::string modelpath,int gpu,int threadnum,int modelsize=640,int maxbatchsize=1);
    ~faceDetectretinaNcnn();
    int Detect(akdData *data,std::vector<std::vector<objBox>> &res);
public:
    int initsuccess=-1;
private:
    void *net=nullptr;
    int netsize=640;
    const float prob_threshold = 0.8f;
    const float nms_threshold = 0.4f;
};

#endif // FACEDETECT_RETINAFACE_NCNN_H

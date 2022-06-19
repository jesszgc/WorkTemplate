#ifndef FACEDETECT_RETINAFACE_TRT_H
#define FACEDETECT_RETINAFACE_TRT_H

#include "../odService.h"

class faceDetectretinaTrt : public odService
{
public:
    faceDetectretinaTrt(std::string modelpath,int gpu,int threadnum,int modelsize=640,int maxbatchsize=1);
    ~faceDetectretinaTrt();
    int Detect(akdData *data,std::vector<std::vector<objBox>> &res);
private:
     int preprocess_cpu(akdData*data,int w,int h);
     int postprocess_cpu(akdData*data,std::vector<std::vector<int>> outputsizes,std::vector<std::vector<objBox>> &res);
public:
    int initsuccess=-1;
private:
    int netsize=640;
    const float prob_threshold = 0.8f;
    const float nms_threshold = 0.4f;

    void *trt=nullptr;
    std::vector<int> maxinputsize;
    std::vector<int> mininputsize;
    std::vector<std::vector<int>> maxoutputsize;

    std::vector<void*> gmm{nullptr};
    std::vector<float*> cmm{nullptr};
    float *input=nullptr;
    float* nms_out=nullptr;

    int inferw=-1;
    int inferh=-1;
};

#endif // FACEDETECT_RETINAFACE_TRT_H

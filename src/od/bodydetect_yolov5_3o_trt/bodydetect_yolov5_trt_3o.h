#ifndef BODYDETECT_YOLOV5_TRT_3O_H
#define BODYDETECT_YOLOV5_TRT_3O_H

#include "../odService.h"
#include<vector>
class bodyDetectYolov5Trt3o : public odService
{
public:
    bodyDetectYolov5Trt3o(std::string modelpath,int gpu,int threadnum,int modelsize=640,int maxbatchsize=1);
    ~bodyDetectYolov5Trt3o();
    int Detect(akdData *data,std::vector<std::vector<objBox>> &res);

public:
    int initsuccess=-1;
private:
    void *trt=nullptr;
    int netsize=640;
    std::vector<int> maxinputsize;
    std::vector<int> mininputsize;
    std::vector<std::vector<int>> maxoutputsize;
    std::vector<void*> gmm{nullptr};
    std::vector<float*> cmm{nullptr};
    float conf_thres_ = 0.3;
    float iou_thres_ = 0.3;
    int numclass=80;
    std::vector<std::vector<std::vector<float>>>anchors_{
        { {10, 13}, {16, 30}, { 33, 23} },
        { {30, 61}, {62, 45}, { 59, 119} },
        { {116, 90},{156,198 },{373, 326} }
    };
    int inferw,inferh;
};

#endif // BODYDETECT_YOLOV5_TRT_H

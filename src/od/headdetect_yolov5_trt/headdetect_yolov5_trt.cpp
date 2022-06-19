#include <float.h>
#include <stdio.h>
#include <vector>
#include<iostream>
#include "headdetect_yolov5_trt.h"
#include "yolov5.h"
static const char* class_names[] = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};
headDetectYolov5Trt::headDetectYolov5Trt(std::string modelpath,int gpu,int threadnum,int modelsize,int maxbatchsize)
{
    OnnxDynamicNetInitParam params;
    params.onnx_model = modelpath;
    //params.rt_model_name = "yolov5.engine";
    params.num_classes = 80;
    std::cout<<"maxbatchsize:"<<maxbatchsize<<std::endl;
    params.maxbatchsize=maxbatchsize;
    params.netsize=modelsize;
    YOLOV5* yolov5=new YOLOV5(params);
    if(yolov5!=nullptr)
    {
        net=yolov5;
        initsuccess=1;
    }
}

headDetectYolov5Trt::~headDetectYolov5Trt() {

    if(net!=nullptr)
    {
        delete (YOLOV5*)net;
        net=nullptr;
    }
}

int headDetectYolov5Trt::Detect(akdData *data,std::vector<std::vector<objBox>> &res)
{

    YOLOV5* yolov5=(YOLOV5*)net;
    std::vector<cv::Mat> ims;
    for(int i=0;i<data->num;++i){
        cv::Mat img(data->heights[i],data->widths[i],CV_8UC3,data->ptr[i]);
        ims.push_back(img);
    }

    std::vector<std::vector<BoxInfo>> pred_boxes =yolov5->Extract(ims);

    for(int i=0;i<pred_boxes.size();++i)
    {
        std::vector<objBox> rettmp;
        for(int j=0;j<pred_boxes[i].size();++j)
        {
            objBox tt;
            tt.prob=pred_boxes[i][j].class_conf;
            tt.label=pred_boxes[i][j].class_idx;
            tt.labelname=class_names[tt.label];
            tt.rect=cvrect(pred_boxes[i][j].x1,pred_boxes[i][j].y1,pred_boxes[i][j].x2-pred_boxes[i][j].x1,pred_boxes[i][j].y2-pred_boxes[i][j].y1);
            rettmp.push_back(tt);
        }
        res.push_back(rettmp);
    }
    return 0;
}


odService* headdetectinit_yolov5_trt(odinitconfig config)
{
    headDetectYolov5Trt * ch=new headDetectYolov5Trt(config.modelpath,config.gpu,config.threadnum,config.netsize,config.maxbatchsize);
    if(ch->initsuccess<0)
    {
        return nullptr;
    }
    return ch;
}


#include <float.h>
#include <stdio.h>
#include <vector>
#include<iostream>
#include "bodydetect_yolov5_trt_3o.h"
#include"3rdwrap/trt/trtService.h"
#include"3rdwrap/cuda/cudaService.h"
#include"opencv2/opencv.hpp"
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
bodyDetectYolov5Trt3o::bodyDetectYolov5Trt3o(std::string modelpath,int gpu,int threadnum,int modelsize,int maxbatchsize)
{
    trtConfig config;
    config.gpuindex=gpu;
    config.modelpath=modelpath;
    config.maxbatchsize=maxbatchsize;
    config.inputmaxsize=modelsize;
    config.inputminsize=modelsize;
    trtIOprofile ioparam;
    trtService* trtnew=new trtService (config,ioparam);
    trtnew->GetIOSize(maxinputsize,mininputsize,maxoutputsize);
    gmm.clear();
    gmm.resize(4);
    long maxinputsizeall=maxinputsize[0]*maxinputsize[1]*maxinputsize[2];
    gmm[0]=akdcuda::safeCudaMalloc(maxinputsizeall*sizeof (float));
    cmm[0]=new float[maxinputsizeall];

    long maxoutputsizeall1=maxoutputsize[0][1]*maxoutputsize[0][1]*maxoutputsize[0][2];
    gmm[1]=akdcuda::safeCudaMalloc(maxoutputsizeall1*sizeof (float));
    cmm[1]=new float[maxoutputsizeall1];

    long maxoutputsizeall2=maxoutputsize[1][1]*maxoutputsize[1][1]*maxoutputsize[1][2];;
    gmm[2]=akdcuda::safeCudaMalloc(maxoutputsizeall2*sizeof (float));
    cmm[2]=new float[maxoutputsizeall2];

    long maxoutputsizeall3=maxoutputsize[2][1]*maxoutputsize[2][1]*maxoutputsize[2][2];;
    gmm[3]=akdcuda::safeCudaMalloc(maxoutputsizeall3*sizeof (float));
    cmm[3]=new float[maxoutputsizeall3];

    trt=trtnew;
    initsuccess=1;
}

bodyDetectYolov5Trt3o::~bodyDetectYolov5Trt3o() {

    if(trt!=nullptr)
    {
        delete (trtService*)trt;
        trt=nullptr;
    }
}

int bodyDetectYolov5Trt3o::Detect(akdData *data,std::vector<std::vector<objBox>> &res)
{
     trtService* runner=(trtService*) trt;
    int num=data->num;
    inferw=640;//data->inferw;
    inferh=640;//data->inferh;
    for(int i=0;i<data->num;++i)
    {
        float* d_in=(float*)gmm[0]+inferw*inferh*3*i;
         cv::Mat img(data->heights[i],data->widths[i],CV_8UC3,data->ptr[i]);
        cv::Mat img_tmp;// = imgs[i].clone();
        img.convertTo(img_tmp,CV_32FC3);
        cv::resize(img_tmp,img_tmp,cv::Size(inferw,inferh));
        img_tmp=img_tmp/255.0;
        std::vector<cv::Mat> channels;
        cv::split(img_tmp, channels);
        int inputcount=inferh*inferw;//input_shape_.count()/3;

        cudaMemcpy((void*)d_in, channels[2].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount), channels[1].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount*2), channels[0].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);

    }

    std::vector<std::vector<int>> outputsizes;
    runner->infer(gmm.data(),num,inferw,inferh,outputsizes);



    return 0;
}


odService* bodydetectinit_yolov5_3o_trt(odinitconfig config)
{
    bodyDetectYolov5Trt3o * ch=new bodyDetectYolov5Trt3o(config.modelpath,config.gpu,config.threadnum,config.maxnetsize,config.maxbatchsize);
    if(ch->initsuccess<0)
    {
        return nullptr;
    }
    return ch;
}


#include"fr_arcface_trt.h"
#include"3rdwrap/trt/trtService.h"
#include"3rdwrap/cuda/cudaService.h"
#include"opencv2/opencv.hpp"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <functional>
frArcfaceTrt::frArcfaceTrt(std::string modelpath,int gpu,int threadnum,int maxbatchsize)
{
    trtConfig config;
    config.gpuindex=gpu;
    config.modelpath=modelpath;
    config.maxbatchsize=maxbatchsize;
    config.inputmaxsize=112;
    config.inputminsize=112;
    trtIOprofile ioparam;
    trtService* trtnew=new trtService (config,ioparam);
    maxinputsize=ioparam.maxinputsize[0];
    maxoutputsize=ioparam.maxoutputsize;

    int nn=1+maxoutputsize.size();
    gmm.clear();
    gmm.resize(nn);
    cmm.clear();
    cmm.resize(nn);

    long maxinputsizeall=maxinputsize[0]*maxinputsize[1]*maxinputsize[2]*maxinputsize[3];
    gmm[0]=akdcuda::safeCudaMalloc(maxinputsizeall*sizeof(float));
    cmm[0]=new float[maxinputsizeall];

    for(int i=1;i<nn;++i){
        long maxoutputsizeall1=1;
        for(int j=0;j<maxoutputsize[i-1].size();++j)
        {
            maxoutputsizeall1*=maxoutputsize[i-1][j];
        }
        gmm[i]=akdcuda::safeCudaMalloc(maxoutputsizeall1*sizeof(float));
        cmm[i]=new float[maxoutputsizeall1];

    }
    trt=trtnew;
    initsuccess=1;
}
frArcfaceTrt::~frArcfaceTrt(){
if(trt!=nullptr)
{
    delete (trtService*)trt;
    trt=nullptr;
}
}
int frArcfaceTrt::preprocess_cpu(akdData*data,int w,int h)
{
    for(int i=0;i<data->num;++i)
    {
        float* d_in=(float*)(gmm[0])+inferw*inferh*3*i;
        cv::Mat img(data->heights[i],data->widths[i],CV_8UC3,data->ptr[i]);
        cv::Mat img_tmp;// = imgs[i].clone();
        img.convertTo(img_tmp, CV_32F);
//        img.convertTo(img_tmp, CV_32F, 1 / 256.f, -0.5);
//        cv::resize(img_tmp,img_tmp,cv::Size(inferw,inferh));
        //img_tmp=img_tmp/255.0;
        std::vector<cv::Mat> channels;
        cv::split(img_tmp, channels);
        int inputcount=inferh*inferw;

        cudaMemcpy((void*)d_in, channels[2].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount), channels[1].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount*2), channels[0].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);

    }
    return -1;
}
int normalizefeat(float *feat,int num)
{
    float fm = 0;
    for (int i = 0; i < num; i++)
    {
        fm += (feat[i] * feat[i]);
    }
    fm = sqrt(fm) + 0.00001;
    for (int i = 0; i < num; i++)
    {
        feat[i] = feat[i] / fm;
    }
    return 0;
}

int frArcfaceTrt::postprocess_cpu(std::vector<std::vector<int>> outputsizes,std::vector<std::vector<float>> &res)
{
    int num=outputsizes[0][0];
    int c=outputsizes[0][1];
    cudaMemcpy(cmm[1],gmm[1],num*c*sizeof (float),cudaMemcpyKind::cudaMemcpyDeviceToHost);
    for(int i=0;i<num;++i)
    {
        float* feat=cmm[1]+512*i;
        normalizefeat(feat,512);
        std::vector<float> v_f(feat,feat+512);
        res.push_back(v_f);
    }
    return -1;
}
int frArcfaceTrt::Extract(akdData *data,std::vector<std::vector<float>> &res){

    trtService* runner=(trtService*) trt;
    // preprocess cpu
    int picnum=data->num;
    if(picnum<=0)
    {
        return 0;
    }
    inferw=112;
    inferh=112;
    double t1=cv::getTickCount();

    preprocess_cpu(data,inferw,inferh);

    double t2=cv::getTickCount();

    std::vector<std::vector<int>> outputsizes;
    runner->infer(gmm.data(),picnum,inferw,inferh,outputsizes);

    double t3=cv::getTickCount();

    postprocess_cpu(outputsizes,res);

    double t4=cv::getTickCount();

    std::cout<<picnum<<"time:"<<(t2-t1)*1000/cv::getTickFrequency()<<" ";
    std::cout<<(t3-t2)*1000/cv::getTickFrequency()<<" ";
    std::cout<<(t4-t3)*1000/cv::getTickFrequency()<<std::endl;
    return 0;
}
frService* frinit_arcface_trt(frinitconfig config)
{
    frArcfaceTrt* fr=new frArcfaceTrt(config.modelpath,config.gpu,config.threadnum,config.maxbatchsize);
    if(fr->initsuccess<1)
    {
        return nullptr;
    }
    return (frService*)fr;
}

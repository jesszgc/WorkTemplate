#include"fr_arcface_ncnn.h"
#include "net.h"
frArcfaceNcnn::frArcfaceNcnn(std::string modelpath)
{
    ncnn::Net *mobilenet=new ncnn::Net();
    mobilenet->opt.use_vulkan_compute = true;
    std::string parampath=modelpath+".param";
    std::string binpath=modelpath+".bin";
    int ret1=mobilenet->load_param(parampath.c_str());//"mnet.25-opt.param");
    int ret2=mobilenet->load_model(binpath.c_str());//"mnet.25-opt.bin");
    if(ret1<0||ret2<0)
    {
        return;
    }
    net=mobilenet;

    initsuccess=1;

}
frArcfaceNcnn::~frArcfaceNcnn(){
    if(net!=nullptr)
    {
        delete (ncnn::Net*)net;
        net=nullptr;
    }
}

int normalizefeat(std::vector<float> &feat)
{
    float fm = 0;
    for (int i = 0; i < feat.size(); i++)
    {
        fm += (feat[i] * feat[i]);
    }
    fm = sqrt(fm) + 0.00001;
    for (int i = 0; i < feat.size(); i++)
    {
        feat[i] = feat[i] / fm;
    }
    return 0;
}
#include"opencv2/opencv.hpp"
int frArcfaceNcnn::Extract(akdData *data,std::vector<std::vector<float>> &res)
{
    ncnn::Net *mobilenet=(ncnn::Net *)net;
    for(int i=0;i<data->num;++i){
        cv::Mat face(112,112,CV_8UC3,data->ptr[i]);
        ncnn::Mat in = ncnn::Mat::from_pixels((unsigned char*)data->ptr[i], ncnn::Mat::PIXEL_BGR2RGB,112, 112);
        ncnn::Extractor ex = mobilenet->create_extractor();
        ex.input("id", in);
        ncnn::Mat out;
        ex.extract("fc1", out);
        out = out.reshape(out.w * out.h * out.c);
        std::vector<float> cls_scores(out.w);
        for (int j = 0; j < out.w; j++)
        {
            cls_scores[j] = out[j];
        }
        normalizefeat(cls_scores);
        res.push_back(cls_scores);
    }
    return 0;
}

frService* frinit_arcface_ncnn(frinitconfig config)
{
    frArcfaceNcnn* fr=new frArcfaceNcnn(config.modelpath);
    if(fr->initsuccess<1)
    {
        return nullptr;
    }
    return (frService*)fr;
}


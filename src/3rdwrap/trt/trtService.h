#ifndef TRTSERVICE_H
#define TRTSERVICE_H
#include<iostream>
#include<string>
#include<vector>
#include"NvInferRuntimeCommon.h"
#include "NvInfer.h"
class Logger : public nvinfer1::ILogger
{
public:
    void log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg)noexcept;
};
enum ModelType{
    onnx=1,
    caffe,
    ncnn_file,
    ncnn_mem,
    tf,
    pth,
    pt
};
enum InferType{
    FP32=1,
    FP16,
    Int8
};

struct trtConfig{
    int maxbatchsize=0;
    std::string modelpath;
    int gpuindex=0;
    int inputmaxsize=0;
    int inputminsize=0;
    ModelType mt=ModelType::onnx;
    InferType it=InferType::FP32;
};
struct trtIOprofile{
std::vector<std::vector<int>> maxinputsize;
std::vector<std::vector<int>> mininputsize;
std::vector<std::vector<int>> maxoutputsize;
};
class trtService{
public:
    trtService(trtConfig &config,trtIOprofile &ioparam);
    ~trtService();
    int infer(void **data,int num,int w,int h,std::vector<std::vector<int>> &outputsizes);
    int GetIOSize(std::vector<int> &maxinputsize,std::vector<int> &mininputsize,std::vector<std::vector<int>> &maxoutputsize);
private:
    cudaStream_t stream_;

    Logger logger_;
    int initsuccess=-1;

    void* runtime_=nullptr;
    void* engine_=nullptr;
    void* context_=nullptr;

    std::vector<int>maxinputsize;
    std::vector<int>mininputsize;

    std::vector<std::vector<int>> maxoutputsize;

};
int deserializeCudaEngine(const void* blob, std::size_t size,void* &runtime_,void* &engine_,void* &context_);
int savetrtmodel(const std::string& path,char* gie_model_stream_,int size);
int readtrtmodel(const std::string& gie_file, std::string &model);
#endif

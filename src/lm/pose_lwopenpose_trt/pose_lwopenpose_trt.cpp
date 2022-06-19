#include"pose_lwopenpose_trt.h"
#include"3rdwrap/trt/trtService.h"
#include"3rdwrap/cuda/cudaService.h"
poseLwOpenposeTrt::poseLwOpenposeTrt(std::string modelpath,int gpu,int threadnum,int modelsize,int maxbatchsize)
{
    trtConfig config;
    config.gpuindex=gpu;
    config.modelpath=modelpath;
    config.maxbatchsize=maxbatchsize;
    config.inputmaxsize=modelsize;
    config.inputminsize=modelsize;
    trtIOprofile ioparam;
    trtService* trtnew=new trtService (config,ioparam);

    gmm.clear();
    gmm.reserve(2);
//    long maxinputsize=5000;
//    gmm[0]=akdcuda::safeCudaMalloc(maxinputsize);
//    cmm[0]=new float[maxinputsize];
//    long maxoutputsize=5000;
//    gmm[1]=akdcuda::safeCudaMalloc(maxoutputsize);
//    cmm[1]=new float[maxoutputsize];
    trt=trtnew;
    initsuccess=1;
}
poseLwOpenposeTrt::~poseLwOpenposeTrt()
{
    if(trt!=nullptr)
    {
        delete (trtService*)trt;
        trt=nullptr;
    }
    for(int i=0;i<gmm.size();++i)
    {
        cudaFree(gmm[i]);
        gmm[i]=nullptr;
    }

    for(int i=0;i<cmm.size();++i)
    {
        delete cmm[i];
        cmm[i]=nullptr;
    }
}
int poseLwOpenposeTrt::preprocess_cpu(akdData *data)
{
    for(int i=0;i<data->num;++i)
    {


    }
    return -1;
}
int poseLwOpenposeTrt::postprocess_cpu(std::vector<std::vector<int>> outputsizes,std::vector<inferOutdata> &res)
{
    //post process cpu
    long outputsize=outputsizes[0][0]*outputsizes[0][1]*outputsizes[0][2];
    cudaMemcpy(cmm[1],gmm[1],outputsize,cudaMemcpyKind::cudaMemcpyDeviceToHost);
//    for(int i=0;i<num;++i)
//    {

//    }
    return -1;
}
int poseLwOpenposeTrt::Infer(akdData *data,std::vector<inferOutdata> &res)
{
    trtService* runner=(trtService*) trt;
    // preprocess cpu
    int num=data->num;
    int w=data->widths[0];
    int h=data->heights[0];
    preprocess_cpu(data);

    std::vector<std::vector<int>> outputsizes;
    runner->infer(gmm.data(),num,w,h,outputsizes);

    postprocess_cpu(outputsizes,res);



    return -1;
}

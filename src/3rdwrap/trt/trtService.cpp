#include"trtService.h"
#include<fstream>
#include<string>
#include<sstream>
#include "parserOnnxConfig.h"
#include"NvCaffeParser.h"
#if _WIN32
#include<io.h>
#else
#include<unistd.h>
#endif
static Logger logger;
int deserializeCudaEngine(const void* blob, std::size_t size,void* runtime_,void* engine_,void* context_)
{
    // 创建运行时
    runtime_ = nvinfer1::createInferRuntime(logger);
    //assert(runtime_ != nullptr);
    // 由运行时根据读取的序列化的模型反序列化生成engine
    engine_ = ((nvinfer1::IRuntime*)runtime_)->deserializeCudaEngine(blob, size);
    //assert(engine_ != nullptr);

    // 利用engine创建执行上下文
    context_ = ((nvinfer1::ICudaEngine* )engine_)->createExecutionContext();
    //assert(context_ != nullptr);

    //mallocInputOutput();
    return 0;
}

int savetrtmodel(const std::string& path,char* gie_model_stream_,int size)
{
    std::ofstream outfile(path, std::ios_base::out | std::ios_base::binary);
    outfile.write(gie_model_stream_, size);
    outfile.close();
    return 0;
}

int readtrtmodel(const std::string& gie_file, std::string &model)
{
    std::ifstream fgie(gie_file, std::ios_base::in | std::ios_base::binary);
    if (!fgie)
        return -1;

    std::stringstream buffer;
    buffer << fgie.rdbuf();

    std::string stream_model(buffer.str());
    model=stream_model;
    //deserializeCudaEngine(stream_model.data(), stream_model.size());

    return 1;
}

trtService::trtService(trtConfig &config,trtIOprofile &ioparam)
{
//    std::fstream check_file(config.modelpath);
//    bool found = check_file.is_open();
//    if (!found)
//    {
//        std::cerr << "model file is not found " << config.modelpath << std::endl;
//        return;
//    }
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
    bool use_fp16_ = builder->platformHasFastFp16()&&(config.it==InferType::FP16);

    std::string trtmodelname=config.modelpath+std::to_string(config.maxbatchsize)+"_"+std::to_string(config.inputminsize)+"_"+std::to_string(config.inputmaxsize);
    if(use_fp16_)
    {
        trtmodelname+="fp16.trt";
        std::cout << "use GPU FP16 !"<< std::endl;
    }else
    {
        trtmodelname+="fp32.trt";
        std::cout << "Using GPU FP32 !" << std::endl;
    }

    if (access(trtmodelname.c_str(), 0))
    {
        //assert(builder != nullptr);
        const auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
        void* parser=nullptr;
        if(config.mt==ModelType::onnx)
        {
            // onnx解析器
            parser = nvonnxparser::createParser(*network, logger);
            bool ret=((nvonnxparser::IParser* )parser)->parseFromFile(config.modelpath.c_str(), 2);
        }
        else if( config.mt==ModelType::caffe) {
            parser=nvcaffeparser1::createCaffeParser();
            std::string prototxt=config.modelpath+".prototxt";
            std::string caffemodel=config.modelpath+".caffemodel";
            ((nvcaffeparser1::ICaffeParser*)parser)->parse(prototxt.c_str(),caffemodel.c_str(),*network,nvinfer1::DataType::kFLOAT);
        }
        nvinfer1::IBuilderConfig* build_config = builder->createBuilderConfig();
        nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
        nvinfer1::ITensor* input = network->getInput(0);

        std::cout << "********************* : " << input->getName() << std::endl;
        nvinfer1::Dims dims = input->getDimensions();
        std::cout << "batchsize: " << dims.d[0] << " channels: " << dims.d[1] << " height: " << dims.d[2] << " width: " << dims.d[3] << std::endl;
        {
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, dims.d[1], config.inputminsize, config.inputminsize });
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ config.maxbatchsize, dims.d[1], config.inputmaxsize, config.inputmaxsize });
            profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ config.maxbatchsize, dims.d[1], config.inputmaxsize, config.inputmaxsize });
            build_config->addOptimizationProfile(profile);
        }
        nvinfer1::ITensor* output = network->getOutput(0);
        std::cout << "********************* : " << output->getName() << std::endl;
        nvinfer1::Dims odims = output->getDimensions();
        std::cout << "batchsize: " << odims.d[0] << " channels: " << odims.d[1] << " height: " << odims.d[2] << " width: " << odims.d[3] << std::endl;

        if (use_fp16_)
        {
            build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        nvinfer1::IHostMemory*  gie_model_stream_ =builder->buildSerializedNetwork(*network,*build_config);// engine->serialize();
        savetrtmodel(trtmodelname,(char*)gie_model_stream_->data(),gie_model_stream_->size());

        delete gie_model_stream_;//->destroy();
        delete (nvonnxparser::IParser* )parser;
        delete network;
    }
    delete builder;

    std::string modelstr;
    bool ret1=readtrtmodel(trtmodelname,modelstr);

    runtime_ = nvinfer1::createInferRuntime(logger);
    //assert(runtime_ != nullptr);
    engine_ = ((nvinfer1::IRuntime*)runtime_)->deserializeCudaEngine(modelstr.data(), modelstr.size());
    //assert(engine_ != nullptr);
    context_ = ((nvinfer1::ICudaEngine* )engine_)->createExecutionContext();
    ((nvinfer1::IExecutionContext*)context_)->setOptimizationProfileAsync(0,stream_);
    //assert(context_ != nullptr);
    int nn = ((nvinfer1::ICudaEngine* )engine_)->getNbBindings();
    for (int i = 0; i < nn; i++)
    {
        if (i == 0)
        {
            nvinfer1::Dims mininDims = ((nvinfer1::ICudaEngine* )engine_)->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);//lpf
            nvinfer1::Dims maxinDims = ((nvinfer1::ICudaEngine* )engine_)->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);//lpf
            int nd = maxinDims.nbDims;
            std::vector<int> maxinputsize,mininputsize;
            for(int j=0;j<nd;++j)
            {
                maxinputsize.push_back(maxinDims.d[j]);
                mininputsize.push_back(mininDims.d[j]);
            }
            ioparam.maxinputsize.push_back(maxinputsize);
            ioparam.mininputsize.push_back(mininputsize);
            ((nvinfer1::IExecutionContext*)context_)->setBindingDimensions(0, maxinDims);
        }
        else
        {
            nvinfer1::Dims outDims = ((nvinfer1::IExecutionContext*)context_)->getBindingDimensions(i);
            std::vector<int> maxoutput;
            int nd = outDims.nbDims;
            for(int j=0;j<nd;++j)
            {
                maxoutput.push_back(outDims.d[j]);
            }
            ioparam.maxoutputsize.push_back(maxoutput);
        }
    }
}

trtService::~trtService()
{
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    if(context_!=nullptr)
    {
        delete (nvinfer1::IExecutionContext*)context_;
        context_=nullptr;
    }
    if(engine_!=nullptr)
    {
        delete (nvinfer1::ICudaEngine*)engine_;
        engine_=nullptr;
    }
    if(runtime_!=nullptr)
    {
        delete (nvinfer1::IRuntime*)runtime_;
        runtime_=nullptr;
    }
}
int trtService::GetIOSize(std::vector<int> &maxinputsize,std::vector<int> &mininputsize,std::vector<std::vector<int>> &maxoutputsize)
{
    maxinputsize.clear();
    mininputsize.clear();
    maxoutputsize.clear();
    int nn = ((nvinfer1::ICudaEngine* )engine_)->getNbBindings();

    for (int i = 0; i < nn; i++)
    {
        if (i == 0)
        {
            nvinfer1::Dims mininDims = ((nvinfer1::ICudaEngine* )engine_)->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);//lpf
            nvinfer1::Dims maxinDims = ((nvinfer1::ICudaEngine* )engine_)->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);//lpf
            int nd = maxinDims.nbDims;
            for(int j=0;j<nd;++j)
            {
                maxinputsize.push_back(maxinDims.d[j]);
                mininputsize.push_back(mininDims.d[j]);
            }
            ((nvinfer1::IExecutionContext*)context_)->setBindingDimensions(0, maxinDims);
        }
        else
        {
            nvinfer1::Dims outDims = ((nvinfer1::IExecutionContext*)context_)->getBindingDimensions(i);
            std::vector<int> maxoutput;
            int nd = outDims.nbDims;
            for(int j=0;j<nd;++j)
            {
                maxoutput.push_back(outDims.d[j]);
            }
            maxoutputsize.push_back(maxoutput);
        }

    }

}
int trtService::infer(void** data,int num,int w,int h,std::vector<std::vector<int>> &outputsizes)
{
    nvinfer1::Dims4 input_dims{ num, 3, h, w };
    ((nvinfer1::IExecutionContext*)context_)->setBindingDimensions(0, input_dims);

    int nn = ((nvinfer1::ICudaEngine* )engine_)->getNbBindings();
    for (int i = 1; i < nn; i++)
    {
        std::vector<int> outputsize;
        nvinfer1::Dims outDims = ((nvinfer1::IExecutionContext*)context_)->getBindingDimensions(i);
        int nd = outDims.nbDims;
        for(int j=0;j<nd;++j)
        {
            outputsize.push_back(outDims.d[j]);
        }
        outputsizes.push_back(outputsize);
    }

   // ((nvinfer1::IExecutionContext*)context_)->enqueueV2(data, stream_, nullptr);

       ((nvinfer1::IExecutionContext*)context_)->executeV2(data);
    //cudaStreamSynchronize(stream_);
    return 0;
}

void Logger::log(nvinfer1::ILogger::Severity severity, nvinfer1::AsciiChar const* msg)noexcept
{
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        std::cerr << "kINTERNAL_ERROR: " << msg << std::endl;
        break;
    case nvinfer1::ILogger::Severity::kERROR:
        std::cerr << "kERROR: " << msg << std::endl;
        break;
    case nvinfer1::ILogger::Severity::kWARNING:
        std::cerr << "kWARNING: " << msg << std::endl;
        break;
    case nvinfer1::ILogger::Severity::kINFO:
        std::cerr << "kINFO: " << msg << std::endl;
        break;
    case nvinfer1::ILogger::Severity::kVERBOSE:
        std::cerr << "kVERBOSE: " << msg << std::endl;
        break;
    default:
        break;
    }
}

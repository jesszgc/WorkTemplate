#include "yolov5.h"
#include <fstream>
#include <sstream>
#include <assert.h>
#include "mat_transform.hpp"
#include "gpu_func.h"
#include "parserOnnxConfig.h"
#include <time.h>
#include <fstream>
#if _WIN32
#include<io.h>
#else
#include<unistd.h>
#endif
int savedata(float* data,std::string path)
{
    ofstream ofs;
    //3、指定打开方式
    ofs.open(path, ios::out);
    //4、写内容
    for(int i=0;i<10000;++i)
    {
        ofs<<data[i]<<std::endl;
    }
    //5、关闭文件
    ofs.close();
}

YOLOV5::YOLOV5(const OnnxDynamicNetInitParam& params) : params_(params)
{
    cudaSetDevice(params.gpu_id);
    //cudaStreamCreate(&stream_);

    out_shape640_.set_no(5 + params.num_classes);
    maxbatchsize=params.maxbatchsize;
    netsize=params.netsize;
    std::string savemodelpath=params.onnx_model +std::to_string(params.maxbatchsize)+".engine";
    std::string modelstr;

    if (access(savemodelpath.c_str(), 0))
    {
        ConvertOnnxModel(params.onnx_model,savemodelpath);
    }

    bool ret1=readtrtmodel(savemodelpath,modelstr);

    runtime_ = nvinfer1::createInferRuntime(logger_);
    assert(runtime_ != nullptr);
    // 由运行时根据读取的序列化的模型反序列化生成engine

    engine_ = ((nvinfer1::IRuntime*)runtime_)->deserializeCudaEngine(modelstr.data(), modelstr.size());
    assert(engine_ != nullptr);
    nvinfer1::Dims dims =engine_->getBindingDimensions(0);
    //engine_->
    //input_shape_.Reshape(maxbatchsize, dims.d[1], dims.d[2], dims.d[3]);
    //std::cout << "engine_ batchsize: " << dims.d[0] << " channels: " << dims.d[1] << " height: " << dims.d[2] << " width: " << dims.d[3] << std::endl;

    // 利用engine创建执行上下文
    context_ = ((nvinfer1::ICudaEngine* )engine_)->createExecutionContext();
//    context_->setOptimizationProfile(0);
    assert(context_ != nullptr);

    int nn = engine_->getNbBindings();

    //int profilenum = engine->getNbOptimizationProfiles();
    nOutputChannel.resize(nn-1);
    nOutputHeight.resize(nn - 1);
    nOutputWidth.resize(nn - 1);
    for (int i = 0; i < nn; i++)
    {
        if (i == 0)
        {
            nvinfer1::Dims inDims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);//lpf

            nvinfer1::Dims mininDims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMIN);//lpf
            nvinfer1::Dims maxinDims = engine_->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);//lpf
            int nd = maxinDims.nbDims;
            if (nd == 4)
            {
                maxbatchsize = inDims.d[0];
                nInputChannel = inDims.d[1];
                nInputHeight = inDims.d[2];
                nInputWidth = inDims.d[3];


                maxbatchsize = maxinDims.d[0];

                nInputChannel = maxinDims.d[1];

                MaxnInputWidth = maxinDims.d[2];
                MaxnInputHeight = maxinDims.d[3];

                MinnInputWidth = mininDims.d[2];
                MinnInputHeight = mininDims.d[3];

            }
            else if (nd == 3)
            {
                maxbatchsize = inDims.d[0];
                nInputHeight = inDims.d[1];
                nInputWidth = inDims.d[2];

                maxbatchsize = maxinDims.d[0];
                MaxnInputWidth = maxinDims.d[1];
                MaxnInputHeight = maxinDims.d[2];
                MinnInputWidth = mininDims.d[1];
                MinnInputHeight = mininDims.d[2];
            }
            else if (nd == 2)
            {
                maxbatchsize = inDims.d[0];
                nInputWidth = inDims.d[1];

                maxbatchsize = maxinDims.d[0];
                MaxnInputWidth = maxinDims.d[1];
                MinnInputWidth = mininDims.d[1];
            }
            else if (nd == 1)
            {
                maxbatchsize = inDims.d[0];

                maxbatchsize = maxinDims.d[0];
            }

            context_->setBindingDimensions(0, inDims);
        }

        else
        {
            nvinfer1::Dims outDims = context_->getBindingDimensions(i);

            int nd = outDims.nbDims;
            if (nd == 4)
            {
                nOutputChannel[i-1] = outDims.d[1];
                nOutputHeight[i-1] = outDims.d[2];
                nOutputWidth[i-1] = outDims.d[3];
            }
            else if (nd == 3)
            {
                nOutputHeight[i-1] = outDims.d[1];
                nOutputWidth[i-1] = outDims.d[2];
            }
            else if (nd == 2)
            {
                nOutputWidth[i-1] = outDims.d[1];
            }
        }

    }


    bool ret2=mallocInputOutput();

}

YOLOV5::~YOLOV5()
{
    //cudaStreamSynchronize(stream_);
    //cudaStreamDestroy(stream_);
   // if (h_input_tensor_ != NULL)
    //    cudaFreeHost(h_input_tensor_);
    if (d_input_tensor_ != NULL)
        cudaFree(d_input_tensor_);
    if(h_output_tensor640_!=nullptr)
    {
        delete h_output_tensor640_;
        h_output_tensor640_=nullptr;
    }
    if(d_output_tensor640_!=nullptr)
    {
        delete d_output_tensor640_;
        d_output_tensor640_=nullptr;
    }
    if(runtime_=nullptr)
    {
        runtime_->destroy();
        runtime_=nullptr;
    }
    if(engine_=nullptr)
    {
        engine_->destroy();
        engine_=nullptr;
    }
    if(context_=nullptr)
    {
        context_->destroy();
        context_=nullptr;
    }


}

//void YOLOV5::ConvertOnnxModel1(const std::string& onnx_file,std::string savemodelpath)
//{
//    std::fstream check_file(onnx_file);
//    bool found = check_file.is_open();
//    if (!found)
//    {
//        std::cerr << "onnx file is not found " << onnx_file << std::endl;
//        exit(0);
//    }

//    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger_);
//    assert(builder != nullptr);

//    const auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
//    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(explicitBatch);
//    builder->setMaxBatchSize(maxbatchsize+10);
//    // onnx解析器
//    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger_);
//    assert(parser->parseFromFile(onnx_file.c_str(), 2));

//    nvinfer1::IBuilderConfig* build_config = builder->createBuilderConfig();
//    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
//    nvinfer1::ITensor* input = network->getInput(0);

//    std::cout << "********************* : " << input->getName() << std::endl;
//    nvinfer1::Dims dims = input->getDimensions();
//    dims.d[0]=maxbatchsize;
//    std::cout << "batchsize: " << dims.d[0] << " channels: " << dims.d[1] << " height: " << dims.d[2] << " width: " << dims.d[3] << std::endl;
//    //std::cout << "maxbatchsize: " << maxbatchsize<<std::endl;

//    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 3, 640, 640 });
//    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ maxbatchsize,3, 640, 640 });
//    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ maxbatchsize, 3, 640, 640 });

//    build_config->addOptimizationProfile(profile);

//    nvinfer1::ITensor* output = network->getOutput(0);
//    std::cout << "********************* : " << output->getName() << std::endl;
//    nvinfer1::Dims odims = output->getDimensions();
//    std::cout << "batchsize: " << odims.d[0] << " channels: " << odims.d[1] << " height: " << odims.d[2] << " width: " << odims.d[3] << std::endl;

//    //build_config->setMaxWorkspaceSize(1 << 30);


//    if (params_.use_fp16_)
//        use_fp16_ = builder->platformHasFastFp16();
//    if (use_fp16_)
//    {
//        build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
//        std::cout << "use FP16jess --" << use_fp16_ << std::endl;
//    }
//    else
//    {
//        std::cout << "Using GPU FP32 !" << std::endl;
//    }

//    nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *build_config);
//    assert(engine != nullptr);
//    //engine->bindingIsInput()
//    gie_model_stream_ = engine->serialize();

//    parser->destroy();
//    engine->destroy();
//    builder->destroy();
//    network->destroy();
//    savetrtmodel(savemodelpath,(char*)gie_model_stream_->data(),gie_model_stream_->size());
//    gie_model_stream_->destroy();

//}
void YOLOV5::ConvertOnnxModel(const std::string& onnx_file,std::string savemodelpath)
{
    std::fstream check_file(onnx_file);
    bool found = check_file.is_open();
    if (!found)
    {
        std::cerr << "onnx file is not found " << onnx_file << std::endl;
        exit(0);
    }
    nvinfer1::IBuilder* pb=nvinfer1::createInferBuilder(logger_);

    assert(pb != nullptr);

    const auto explicitBatch = 1U << static_cast<uint32_t> (nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* pn = pb->createNetworkV2(explicitBatch);

    // onnx解析器
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*pn, logger_);
    assert(parser->parseFromFile(onnx_file.c_str(), 2));
    pb->setMaxBatchSize(maxbatchsize);

      //pb->setMaxWorkspaceSize(MAX_WORKSPACE);
    nvinfer1::IBuilderConfig* build_config = pb->createBuilderConfig();
    build_config->setMaxWorkspaceSize(1<<30);
    nvinfer1::IOptimizationProfile* profile = pb->createOptimizationProfile();
    nvinfer1::ITensor* input = pn->getInput(0);

    nvinfer1::Dims dims = input->getDimensions();
    dims.d[0]=maxbatchsize;
    std::cout << "*******:" << input->getName()<< " " << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << " " << dims.d[3] << std::endl;

    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 3,netsize , netsize });
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 5,3, netsize, netsize });
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ maxbatchsize, 3, netsize, netsize });
    build_config->addOptimizationProfile(profile);

    nvinfer1::ITensor* output = pn->getOutput(0);
    nvinfer1::Dims odims = output->getDimensions();
    odims.d[0]=maxbatchsize;
    std::cout << "*******:" << output->getName() << " " << odims.d[0] << " " << odims.d[1] << " " << odims.d[2] << " " << odims.d[3] << std::endl;

    //build_config->setMaxWorkspaceSize(1 << 30);
    if (params_.use_fp16_)
        use_fp16_ = pb->platformHasFastFp16();
    if (use_fp16_)
    {
        build_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "use FP16jess --" << use_fp16_ << std::endl;
    }
    nvinfer1::ICudaEngine* engine = pb->buildEngineWithConfig(*pn, *build_config);
    //nvinfer1::ICudaEngine* engine = pb->buildEngineWithConfig(*pn, *build_config);
    //gie_model_stream_=pb->buildSerializedNetwork(*pn, *build_config);
    assert(engine != nullptr);
    gie_model_stream_ = engine->serialize();

    parser->destroy();
     pn->destroy();
    engine->destroy();


    pb->destroy();
    savetrtmodel(savemodelpath,(char*)gie_model_stream_->data(),gie_model_stream_->size());
    gie_model_stream_->destroy();

}


inline void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem = nullptr;
    cudaError cudaerror = cudaMalloc(&deviceMem, memSize);
    if (deviceMem == nullptr || cudaerror != cudaSuccess)
    {
        std::cout << "cuda Out of memory" << std::endl;
        exit(-1);
    }
    return deviceMem;
}
bool YOLOV5::mallocInputOutput()
{
    buffers_.clear();
    int inputcount=input_shape_.count();
    //cudaHostAlloc((void**)&h_input_tensor_, inputcount *maxbatchsize* sizeof(float), cudaHostAllocDefault);  // 3 * 640 * 640
    //cudaMalloc((void**)&d_input_tensor_, inputcount*maxbatchsize * sizeof(float)); // 3 * 640 * 640
    buffers_[0] = safeCudaMalloc(inputcount*maxbatchsize * sizeof(float));

    int outputsizeall=outputsize[0]*outputsize[1]*outputsize[2];
    //cudaHostAlloc((void**)&h_output_tensor640_, outputsizeall *maxbatchsize* sizeof(float), cudaHostAllocDefault);  // 3 * 20 * 20 * 6
    h_output_tensor640_ = new float[outputsizeall*maxbatchsize];
    //cudaMalloc((void**)&d_output_tensor640_, outputsizeall*maxbatchsize * sizeof(float));  // 3 * 20 * 20 * 6
    buffers_[1] = safeCudaMalloc(outputsizeall*maxbatchsize * sizeof(float));
    //std::cout<<"inputcount "<<inputcount<<"outputsizeall "<<outputsizeall<<std::endl;
   // buffers_[0]=(d_input_tensor_);
    //buffers_[1]=(d_output_tensor640_);

    return true;
}


#include <chrono>
std::vector<std::vector<BoxInfo>> YOLOV5::Extract(std::vector<cv::Mat>& img)
{
    //    if (img.size()==0)
    //        return {};

 //   double t1=cv::getTickCount();
auto t11 = std::chrono::high_resolution_clock::now();
    PreprocessCPU(img);
auto t22 = std::chrono::high_resolution_clock::now();
 //   double t2=cv::getTickCount();

    Forward(img.size());
auto t33 = std::chrono::high_resolution_clock::now();
 //   double t3=cv::getTickCount();

    std::vector<std::vector<BoxInfo>> pred_boxes = PostprocessCPU_batch(img.size());
    //coord_scale(img, pred_boxes);
 //   double t4=cv::getTickCount();
auto t44 = std::chrono::high_resolution_clock::now();
//    std::cout<<"preprocess time: "<<(t2-t1)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
//    std::cout<<"Forward time: "<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
//    std::cout<<"postprocess time: "<<(t4-t3)*1000/cv::getTickFrequency()<<"ms"<<std::endl<<std::endl;

    std::cout<<"preprocess time: "<<(( std::chrono::duration<double, std::milli>)(t22-t11)).count()<<"ms"<<std::endl;
    std::cout<<"Forward time: "<<(( std::chrono::duration<double, std::milli>)(t33-t22)).count()<<"ms"<<std::endl;
    std::cout<<"postprocess time: "<<(( std::chrono::duration<double, std::milli>)(t44-t33)).count()<<"ms"<<std::endl<<std::endl;
    return pred_boxes;//std::vector<std::vector<BoxInfo>>{pred_boxes};
}

void YOLOV5::Forward(int num)
{

    nvinfer1::Dims4 dimsIn;
        dimsIn.nbDims = 4;
        dimsIn.d[0] = num;
        dimsIn.d[1] =3;
        dimsIn.d[2] = 640;
        dimsIn.d[3] = 640;
    //nvinfer1::Dims4 input_dims{ num, input_shape_.channels(), input_shape_.height(), input_shape_.width() };
    // std::cout<<"setBindingDimensions num :"<<num<<std::endl;
    // context_->setOptimizationProfile(0);
    context_->setBindingDimensions(0, dimsIn);
   //bool bEXE = context_->execute(num, &buffers_[0]);
   bool bEXE = context_->executeV2(buffers_.data());
   if(bEXE!=true)
   {
        std::cout<<"execute fail"<<std::endl;
   }
   //nvinfer1::Dims dd=context_->getBindingDimensions(0);
    //std::cout<<"BindingDimensions 0 :"<<dd.d[0]<<" "<<dd.d[1]<<" "<<dd.d[2]<<" "<<dd.d[3]<<std::endl;

    //nvinfer1::Dims dd1=context_->getBindingDimensions(1);
    //std::cout<<"BindingDimensions 1 :"<<dd1.d[0]<<" "<<dd1.d[1]<<" "<<dd1.d[2]<<" "<<dd1.d[3]<<std::endl;
   // context_->enqueueV2(buffers_.data(), stream_, nullptr);
   // context_->executeV2(buffers_.data());
    //cudaStreamSynchronize(stream_);
}

void YOLOV5::PreprocessCPU(std::vector<cv::Mat>& imgs)
{
    for(int i=0;i<imgs.size();++i){

        float* d_in=&((float*)buffers_[0])[input_shape_.count()*i];

        cv::Mat img_tmp = imgs[i].clone();
        //cv::imshow("dsdsds",img_tmp);
        img_tmp.convertTo(img_tmp,CV_32FC3);
        cv::resize(img_tmp,img_tmp,cv::Size(640,640));
        img_tmp=img_tmp/255.0;
        std::vector<cv::Mat> channels;
        cv::split(img_tmp, channels);
        int inputcount=input_shape_.count()/3;
        cudaMemcpy((void*)d_in, channels[2].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount), channels[1].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy((void*)(d_in+inputcount*2), channels[0].data, inputcount * sizeof(float), cudaMemcpyHostToDevice);
        //img_tmp.release();
    }
}


std::vector<std::vector<BoxInfo>> YOLOV5::PostprocessCPU_batch(int num)
{
    //std::cout << "---------- PostprocessCPU ----------------------" << std::endl;
    //std::cout<<"out_shape640_.count():"<<out_shape640_.count()<<std::endl;
    int outputsizeall=outputsize[0]*outputsize[1]*outputsize[2]*num;
    cudaMemcpy(h_output_tensor640_, buffers_[1], outputsizeall*num*sizeof(float), cudaMemcpyDeviceToHost);


    filted_pred_boxes_.clear();
    DecodeBoxesCPU_all(h_output_tensor640_, out_shape640_.channels(), out_shape640_.height(), out_shape640_.width(),num);

    //cout << "filted_pred_boxes_ size: " << filted_pred_boxes_.size() << endl;

    vector<vector<BoxInfo>> pred_boxes = NMS(num);
    //cout << "pred boxes size: " << pred_boxes.size() << endl;
    return pred_boxes;
}

void YOLOV5::DecodeBoxesCPU_all(float* ptr, int channels, int height, int width,int num)
{
    int no = numclass+5;
    for(int index=0;index<num;++index){
        std::vector<BoxInfo> bb;
        for (int c=0;c<outputsize[1]; ++c )
        {
            int offset = c  * no ;

            float w=ptr[offset + 2];
            float h=ptr[offset + 3];
            float cx=ptr[offset + 0]-w/2;
            float cy=ptr[offset + 1]-h/2;
            float obj_conf = ptr[offset + 4];

            if (obj_conf <= conf_thres_)
                continue;


            float class_conf;


            float* pbegin = ptr + offset + 5;
            float* pend = pbegin + numclass;
            int class_label = std::max_element(pbegin, pend) - pbegin;
            class_conf=pbegin[class_label];

            float score = class_conf * obj_conf;
            if (score <= conf_thres_)
                continue;

            BoxInfo box(cx,cy,cx+w,cy+h,
                        class_conf, score, class_label);

            bb.emplace_back(box);
        }
        filted_pred_boxes_.emplace_back(bb);

    }

}

vector<vector<BoxInfo>> YOLOV5::NMS(int num)
{
    vector< vector<BoxInfo>> boxes(num);
    for(int ii=0;ii<num;++ii){
        vector<BoxInfo> pred_boxes;
        if (filted_pred_boxes_[ii].empty())
        {
            boxes[ii]={};
        }
        //return {};//pred_boxes;

        sort(filted_pred_boxes_[ii].begin(), filted_pred_boxes_[ii].end(), compose);

        RefineBoxes();
        char* removed = (char*)malloc(filted_pred_boxes_[ii].size() * sizeof(char));
        memset(removed, 0, filted_pred_boxes_[ii].size() * sizeof(char));
        for (int i = 0; i < filted_pred_boxes_[ii].size(); i++)
        {
            if (removed[i])
                continue;

            pred_boxes.push_back(filted_pred_boxes_[ii][i]);
            for (int j = i + 1; j < filted_pred_boxes_[ii].size(); j++)
            {
                if (filted_pred_boxes_[ii][i].class_idx != filted_pred_boxes_[ii][j].class_idx)
                    continue;
                float iou = IOU(filted_pred_boxes_[ii][i], filted_pred_boxes_[ii][j]);
                if (iou >= iou_thres_)
                    removed[j] = 1;
            }
        }
        boxes[ii]=pred_boxes;
        //return std::move(pred_boxes);
    }
    return boxes;
}

// 调整预测框，使框的值处于合理范围
void YOLOV5::RefineBoxes()
{
    for (auto& boxs : filted_pred_boxes_) {


        for (auto& box : boxs)
        {
            box.x1 = box.x1 < 0. ? 0. : box.x1;
            box.x1 = box.x1 > 640. ? 640. : box.x1;
            box.y1 = box.y1 < 0. ? 0. : box.y1;
            box.y1 = box.y1 > 640. ? 640. : box.y1;
            box.x2 = box.x2 < 0. ? 0. : box.x2;
            box.x2 = box.x2 > 640. ? 640. : box.x2;
            box.y2 = box.y2 < 0. ? 0. : box.y2;
            box.y2 = box.y2 > 640. ? 640. : box.y2;
        }}
}

float YOLOV5::IOU(BoxInfo& b1, BoxInfo& b2)
{
    float x1 = b1.x1 > b2.x1 ? b1.x1 : b2.x1;
    float y1 = b1.y1 > b2.y1 ? b1.y1 : b2.y1;
    float x2 = b1.x2 < b2.x2 ? b1.x2 : b2.x2;
    float y2 = b1.y2 < b2.y2 ? b1.y2 : b2.y2;

    float inter_area = ((x2 - x1) < 0 ? 0 : (x2 - x1)) * ((y2 - y1) < 0 ? 0 : (y2 - y1));
    float b1_area = (b1.x2 - b1.x1) * (b1.y2 - b1.y1);
    float b2_area = (b2.x2 - b2.x1) * (b2.y2 - b2.y1);

    return inter_area / (b1_area + b2_area - inter_area + 1e-5);
}

void YOLOV5::coord_scale(const cv::Mat& img, vector<BoxInfo>& pred_boxes)
{
    int h = int(round(img.rows * rate_));
    int w = int(round(img.cols * rate_));

    int dw = (crop_size_.width - w) % 32;
    int dh = (crop_size_.height - h) % 32;
    float fdw = dw / 2.;
    float fdh = dh / 2.;

    int top = int(round(fdh - 0.1));
    int left = int(round(fdw - 0.1));

    for (auto& box : pred_boxes)
    {
        box.x1 = (box.x1 - left) / rate_;
        box.x2 = (box.x2 - left) / rate_;
        box.y1 = (box.y1 - top)/ rate_;
        box.y2 = (box.y2 - top) / rate_;
    }
}

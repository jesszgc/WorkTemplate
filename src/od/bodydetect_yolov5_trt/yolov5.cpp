#include "yolov5.h"
#include <fstream>
#include <sstream>
#include <assert.h>
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
    cudaError cudaerror =cudaStreamCreate(&stream_);
    if (cudaerror != cudaSuccess)
    {
        std::cout << "cudaStreamCreate fail"<<cudaerror << std::endl;
        exit(-1);
    }
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
    //runtime_->setDLACore(1);

    engine_ = ((nvinfer1::IRuntime*)runtime_)->deserializeCudaEngine(modelstr.data(), modelstr.size());
    assert(engine_ != nullptr);
    // 利用engine创建执行上下文
    context_ = ((nvinfer1::ICudaEngine* )engine_)->createExecutionContext();
    context_->setOptimizationProfileAsync(0,stream_);

    assert(context_ != nullptr);

    int nn = engine_->getNbBindings();

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
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    if(h_output_tensor640_!=nullptr)
    {
        delete []h_output_tensor640_;
        h_output_tensor640_=nullptr;
    }

    if(context_==nullptr)
    {
        delete context_;
        context_=nullptr;
    }
    if(engine_==nullptr)
    {
        delete engine_;
        engine_=nullptr;
    }
    if(runtime_==nullptr)
    {
        delete runtime_;
        runtime_=nullptr;
    }
}


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
    //pb->setMaxBatchSize(maxbatchsize);

    nvinfer1::IBuilderConfig* build_config = pb->createBuilderConfig();
//    build_config->setMaxWorkspaceSize(6<<30);
    nvinfer1::IOptimizationProfile* profile = pb->createOptimizationProfile();
    nvinfer1::ITensor* input = pn->getInput(0);

    nvinfer1::Dims dims = input->getDimensions();
    //dims.d[0]=-1;
    //input->setDimensions(dims);
    std::cout << "inputsize: " << input->getName()<< " " << dims.d[0] << " " << dims.d[1] << " " << dims.d[2] << " " << dims.d[3] << std::endl;

    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 3,netsize , netsize });
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ maxbatchsize,3, netsize, netsize });
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ maxbatchsize, 3, netsize, netsize });
    build_config->addOptimizationProfile(profile);


    nvinfer1::IHostMemory*  gie_model_stream_=pb->buildSerializedNetwork(*pn, *build_config);
    //assert(engine != nullptr);
    //gie_model_stream_ = engine->serialize();

    delete parser;//->destroy();
    delete pn;//->destroy();
    //engine->destroy();
    delete build_config;

    delete pb;//->destroy();
    savetrtmodel(savemodelpath,(char*)gie_model_stream_->data(),gie_model_stream_->size());
    delete gie_model_stream_;//->destroy();

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
    buffers_.resize(2,nullptr);
    int inputcount=input_shape_.count();
    //cudaHostAlloc((void**)&h_input_tensor_, inputcount *maxbatchsize* sizeof(float), cudaHostAllocDefault);  // 3 * 640 * 640
    //cudaMalloc((void**)&d_input_tensor_, inputcount*maxbatchsize * sizeof(float)); // 3 * 640 * 640
    buffers_[0] = safeCudaMalloc(inputcount*maxbatchsize * sizeof(float));

    int outputsizeall=outputsize[0]*outputsize[1]*outputsize[2];
    //cudaHostAlloc((void**)&h_output_tensor640_, outputsizeall *maxbatchsize* sizeof(float), cudaHostAllocDefault);  // 3 * 20 * 20 * 6
    h_output_tensor640_ = new float[outputsizeall*maxbatchsize];
    //h_output_tensor640_[0]=-20;
    //cudaMalloc((void**)&d_output_tensor640_, outputsizeall*maxbatchsize * sizeof(float));  // 3 * 20 * 20 * 6
    buffers_[1] = safeCudaMalloc(outputsizeall*maxbatchsize * sizeof(float));
    buffers_[2] = safeCudaMalloc(outputsizeall*maxbatchsize * sizeof(float));
    buffers_[3] = safeCudaMalloc(outputsizeall*maxbatchsize * sizeof(float));
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

    std::cout<<img.size()<<" pre: "<<(( std::chrono::duration<double, std::milli>)(t22-t11)).count()<<"ms"<<std::endl;
    std::cout<<img.size()<<" Forward: "<<(( std::chrono::duration<double, std::milli>)(t33-t22)).count()<<"ms"<<std::endl;
    std::cout<<img.size()<<" post: "<<(( std::chrono::duration<double, std::milli>)(t44-t33)).count()<<"ms"<<std::endl<<std::endl;
    return pred_boxes;//std::vector<std::vector<BoxInfo>>{pred_boxes};
}

void YOLOV5::Forward(int num)
{

    nvinfer1::Dims4 dimsIn;
    dimsIn.nbDims = 4;
    dimsIn.d[0] = 2;
    dimsIn.d[1] =3;
    dimsIn.d[2] = 640;
    dimsIn.d[3] = 640;
    //nvinfer1::Dims4 input_dims{ num, input_shape_.channels(), input_shape_.height(), input_shape_.width() };
    // std::cout<<"setBindingDimensions num :"<<num<<std::endl;
    // context_->setOptimizationProfile(0);
    // context_->setInputShapeBinding();
    context_->setBindingDimensions(0, dimsIn);

    std::vector<std::vector<int>> outputsizes;
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



    //bool bEXE = context_->execute(num, &buffers_[0]);
    if (!context_->allInputDimensionsSpecified())
    {
        std::cout<<"allInputDimensionsSpecified fail"<<std::endl;
        //return false;
    }
    //bool bEXE = context_->execute(num,buffers_.data());
    // double t1=cv::getTickCount();
    //bool bEXE =context_->enqueueV2(buffers_.data(), stream_, nullptr);
    bool bEXE = context_->executeV2(buffers_.data());
    //double t2=cv::getTickCount();
    //std::cout<<num<<" context_->executeV2(buffers_.data())"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
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
    cudaStreamSynchronize(stream_);
}

void YOLOV5::PreprocessCPU(std::vector<cv::Mat>& imgs)
{
    for(int i=0;i<imgs.size();++i){

        // std::cout<<"PreprocessCPU buffers_.size()"<<buffers_.size()<<std::endl;
        //float* d_in=&((float*)buffers_[0])[input_shape_.count()*i];
        float* d_in=(float*)buffers_[0]+640*640*3*i;
        cv::Mat img_tmp;// = imgs[i].clone();
        //cv::imshow("dsdsds",img_tmp);
        imgs[i].convertTo(img_tmp,CV_32FC3);
        cv::resize(img_tmp,img_tmp,cv::Size(640,640));
        img_tmp=img_tmp/255.0;
        std::vector<cv::Mat> channels;
        cv::split(img_tmp, channels);
        int inputcount=640*640;//input_shape_.count()/3;
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
    int outputsizeall=outputsize[0]*outputsize[1]*outputsize[2];
    //float* oo=new float[outputsizeall*num];
    cudaError cudaerror=cudaMemcpy((void*)h_output_tensor640_, buffers_[1], outputsizeall*num*sizeof(float), cudaMemcpyDeviceToHost);
    if ( cudaerror != cudaSuccess)
    {
        std::cout <<num <<" cuda Out of memory" <<cudaerror<<std::endl;
        exit(-1);
    }
    //std::cout<<"PostprocessCPU_batch buffers_.size()"<<buffers_.size()<<std::endl;
    //memcpy(h_output_tensor640_,oo,outputsizeall*num);
    //delete[] oo;
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
        float *pp=ptr+(no*outputsize[1])*index;
        std::vector<BoxInfo> bb;
        for (int c=0;c<outputsize[1]; ++c )
        {
            int offset = c  * no ;

            float w=pp[offset + 2];
            float h=pp[offset + 3];
            float cx=pp[offset + 0]-w/2;
            float cy=pp[offset + 1]-h/2;
            float obj_conf = pp[offset + 4];

            if (obj_conf <= conf_thres_)
                continue;


            float class_conf;


            float* pbegin = pp + offset + 5;
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

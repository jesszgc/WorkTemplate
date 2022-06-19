#ifndef YOLOV5_H_
#define YOLOV5_H_

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "NvOnnxParser.h"
#include"NvInferRuntimeCommon.h"
#include "NvInfer.h"
#include "common.hpp"
#include <mutex>
#include"3rdwrap/trt/trtService.h"
using namespace std;

struct OnnxDynamicNetInitParam
{
    std::string onnx_model;
    int gpu_id = 0;
//    std::string rt_stream_path = "./";
//    std::string rt_model_name = "defaule.gie";
    bool use_fp16_ = true;
    int num_classes;
    int maxbatchsize=1;
    int netsize=640;
};
class YOLOV5
{
public:
    //YOLOV5() = delete;
    YOLOV5(const OnnxDynamicNetInitParam& param);
    ~YOLOV5();
    std::vector<std::vector<BoxInfo>> Extract(std::vector<cv::Mat>& img);
private:
    // 直接加载onnx模型，并转换成trt模型
    void ConvertOnnxModel(const std::string& onnx_file,std::string savemodelpath);
    void SaveRtModel(const std::string& path);
    bool mallocInputOutput();
private:
    void Forward(int num);
    void PreprocessCPU(std::vector< cv::Mat>& imgs);
    std::vector<std::vector<BoxInfo>> PostprocessCPU_batch(int num);
    void DecodeBoxesCPU_all(float* ptr, int channels, int height, int width,int num);
    std::vector<vector<BoxInfo>> NMS(int num);
   static bool compose(BoxInfo& box1, BoxInfo& box2)
    {
        return box1.score > box2.score;
    }
    // 调整预测框，使框的值处于合理范围
    inline void RefineBoxes();
    inline float IOU(BoxInfo& b1, BoxInfo& b2);
    void coord_scale(const cv::Mat &img, vector<BoxInfo>& pred_boxes);
private:
    OnnxDynamicNetInitParam params_;
    cudaStream_t stream_;

    Logger logger_;
    bool use_fp16_;

    nvinfer1::IRuntime* runtime_=nullptr;
    nvinfer1::ICudaEngine* engine_=nullptr;
    nvinfer1::IExecutionContext* context_=nullptr;
    //nvinfer1::IHostMemory* gie_model_stream_{ nullptr };
//jess
    int maxbatchsize=1;

    int nInputChannel;
    int nInputHeight;
    int nInputWidth;

    int MinnInputHeight;
    int MinnInputWidth;

    int MaxnInputHeight;
    int MaxnInputWidth;

    std::vector<int> nOutputChannel;
    std::vector<int> nOutputHeight;
    std::vector<int> nOutputWidth;
    int netsize=640;
    cv::Size crop_size_{ 640, 640 };

    Shape input_shape_{1, 3, 640, 640};
    Shape out_shape640_{ 1, 3, 640, 640, 5+80 };
    int numclass=80;
    std::vector<int> outputsize{1,25200,85};

    vector<vector<vector<float>>> anchors_{
        { {10, 13}, {16, 30}, { 33, 23} },
        { {30, 61}, {62, 45}, { 59, 119} },
        { {116, 90},{156,198 },{373, 326} }
    };

    float conf_thres_ = 0.3;
    float iou_thres_ = 0.3;

    std::vector<vector<BoxInfo>> filted_pred_boxes_;
    float* h_output_tensor640_=nullptr;
    vector<void*> buffers_{nullptr,nullptr};
    float rate_;
    std::mutex mtx_;
};

#endif

#include <float.h>
#include <stdio.h>
#include <vector>
#include<iostream>
#include "facedetect_retinaface_trt.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>


//static inline float intersection_area( objBox& a,  objBox& b)
//{
//    //cv::Rect_<float> inter
//    cvrect r= a.rect & b.rect;
//    return r.area();
//}

//static void qsort_descent_inplace(std::vector<objBox>& faceobjects, int left, int right)
//{
//    int i = left;
//    int j = right;
//    float p = faceobjects[(left + right) / 2].prob;

//    while (i <= j)
//    {
//        while (faceobjects[i].prob > p)
//            i++;

//        while (faceobjects[j].prob < p)
//            j--;

//        if (i <= j)
//        {
//            // swap
//            std::swap(faceobjects[i], faceobjects[j]);

//            i++;
//            j--;
//        }
//    }

//#pragma omp parallel sections
//    {
//#pragma omp section
//        {
//            if (left < j) qsort_descent_inplace(faceobjects, left, j);
//        }
//#pragma omp section
//        {
//            if (i < right) qsort_descent_inplace(faceobjects, i, right);
//        }
//    }
//}

//static void qsort_descent_inplace(std::vector<objBox>& faceobjects)
//{
//    if (faceobjects.empty())
//        return;

//    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
//}

//static void nms_sorted_bboxes(std::vector<objBox>& faceobjects, std::vector<int>& picked, float nms_threshold)
//{
//    picked.clear();

//    const int n = faceobjects.size();

//    std::vector<float> areas(n);
//    for (int i = 0; i < n; i++)
//    {
//        areas[i] = faceobjects[i].rect.area();
//    }

//    for (int i = 0; i < n; i++)
//    {
//        objBox& a = faceobjects[i];

//        int keep = 1;
//        for (int j = 0; j < (int)picked.size(); j++)
//        {
//            objBox& b = faceobjects[picked[j]];

//            // intersection over union
//            float inter_area = intersection_area(a, b);
//            float union_area = areas[i] + areas[picked[j]] - inter_area;
//            //             float IoU = inter_area / union_area
//            if (inter_area / union_area > nms_threshold)
//                keep = 0;
//        }

//        if (keep)
//            picked.push_back(i);
//    }
//}

//// copy from src/layer/proposal.cpp
//static ncnn::Mat generate_anchors(int base_size, const ncnn::Mat& ratios, const ncnn::Mat& scales)
//{
//    int num_ratio = ratios.w;
//    int num_scale = scales.w;

//    ncnn::Mat anchors;
//    anchors.create(4, num_ratio * num_scale);

//    const float cx = base_size * 0.5f;
//    const float cy = base_size * 0.5f;

//    for (int i = 0; i < num_ratio; i++)
//    {
//        float ar = ratios[i];

//        int r_w = round(base_size / sqrt(ar));
//        int r_h = round(r_w * ar); //round(base_size * sqrt(ar));

//        for (int j = 0; j < num_scale; j++)
//        {
//            float scale = scales[j];

//            float rs_w = r_w * scale;
//            float rs_h = r_h * scale;

//            float* anchor = anchors.row(i * num_scale + j);

//            anchor[0] = cx - rs_w * 0.5f;
//            anchor[1] = cy - rs_h * 0.5f;
//            anchor[2] = cx + rs_w * 0.5f;
//            anchor[3] = cy + rs_h * 0.5f;
//        }
//    }

//    return anchors;
//}

//static void generate_proposals(const ncnn::Mat& anchors, int feat_stride, const ncnn::Mat& score_blob, const ncnn::Mat& bbox_blob, const ncnn::Mat& landmark_blob, float prob_threshold, std::vector<objBox>& faceobjects)
//{
//    int w = score_blob.w;
//    int h = score_blob.h;

//    // generate face proposal from bbox deltas and shifted anchors
//    const int num_anchors = anchors.h;

//    for (int q = 0; q < num_anchors; q++)
//    {
//        const float* anchor = anchors.row(q);

//        const ncnn::Mat score = score_blob.channel(q + num_anchors);
//        const ncnn::Mat bbox = bbox_blob.channel_range(q * 4, 4);
//        const ncnn::Mat landmark = landmark_blob.channel_range(q * 10, 10);

//        // shifted anchor
//        float anchor_y = anchor[1];

//        float anchor_w = anchor[2] - anchor[0];
//        float anchor_h = anchor[3] - anchor[1];

//        for (int i = 0; i < h; i++)
//        {
//            float anchor_x = anchor[0];

//            for (int j = 0; j < w; j++)
//            {
//                int index = i * w + j;

//                float prob = score[index];

//                if (prob >= prob_threshold)
//                {
//                    // apply center size
//                    float dx = bbox.channel(0)[index];
//                    float dy = bbox.channel(1)[index];
//                    float dw = bbox.channel(2)[index];
//                    float dh = bbox.channel(3)[index];

//                    float cx = anchor_x + anchor_w * 0.5f;
//                    float cy = anchor_y + anchor_h * 0.5f;

//                    float pb_cx = cx + anchor_w * dx;
//                    float pb_cy = cy + anchor_h * dy;

//                    float pb_w = anchor_w * exp(dw);
//                    float pb_h = anchor_h * exp(dh);

//                    float x0 = pb_cx - pb_w * 0.5f;
//                    float y0 = pb_cy - pb_h * 0.5f;
//                    float x1 = pb_cx + pb_w * 0.5f;
//                    float y1 = pb_cy + pb_h * 0.5f;

//                    objBox obj;
//                    obj.rect.x = x0;
//                    obj.rect.y = y0;
//                    obj.rect.width = x1 - x0 + 1;
//                    obj.rect.height = y1 - y0 + 1;
//                    obj.pp.clear();
//                    for(int jj=0;jj<10;jj+=2)
//                    {
//                        int xx=cx + (anchor_w + 1) * landmark.channel(jj)[index];
//                        int yy=cy + (anchor_h + 1) * landmark.channel(jj+1)[index];
//                        obj.pp.push_back(cvpoint(xx,yy));

//                    }

//                    obj.prob = prob;

//                    faceobjects.push_back(obj);
//                }

//                anchor_x += feat_stride;
//            }

//            anchor_y += feat_stride;
//        }
//    }
//}

#include"3rdwrap/trt/trtService.h"
#include"3rdwrap/cuda/cudaService.h"
faceDetectretinaTrt::faceDetectretinaTrt(std::string modelpath,int gpu,int threadnum,int modelsize,int maxbatchsize)
{
    trtConfig config;
    config.gpuindex=gpu;
    config.modelpath=modelpath;
    config.maxbatchsize=maxbatchsize;
    config.inputmaxsize=modelsize;
    config.inputminsize=modelsize/2;
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

faceDetectretinaTrt::~faceDetectretinaTrt() {

    if(trt!=nullptr)
    {
        delete (trtService*)trt;
        trt=nullptr;
    }
}
int faceDetectretinaTrt::preprocess_cpu(akdData*data,int w,int h){
    for(int i=0;i<data->num;++i)
    {
        float* d_in=(float*)(gmm[0])+inferw*inferh*3*i;
        cv::Mat img(data->heights[i],data->widths[i],CV_8UC3,data->ptr[i]);
        cv::Mat img_tmp;// = imgs[i].clone();
//        img.convertTo(img_tmp, CV_32F);
        img.convertTo(img_tmp, CV_32F, 1 / 256.f, -0.5);
        cv::resize(img_tmp,img_tmp,cv::Size(inferw,inferh));
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
int faceDetectretinaTrt::postprocess_cpu(akdData*data,std::vector<std::vector<int>> outputsizes,std::vector<std::vector<objBox>> &res){
for(int i=0;i<outputsizes.size();++i)
{
    int s=outputsizes[i][0]*outputsizes[i][1]*outputsizes[i][2];
    cudaMemcpy(cmm[i+1],gmm[i+1],s*sizeof (float),cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

//    for(int i=0;i<data->num;++i){
//        int img_w = data->widths[i];//.cols;
//        int img_h = data->heights[i];//bgr.rows;


//        std::vector<objBox> faceproposals;

//        // stride 32
//        {
//            ncnn::Mat score_blob, bbox_blob, landmark_blob;

//            const int base_size = 16;
//            const int feat_stride = 32;
//            ncnn::Mat ratios(1);
//            ratios[0] = 1.f;
//            ncnn::Mat scales(2);
//            scales[0] = 32.f;
//            scales[1] = 16.f;
//            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

//            std::vector<objBox> faceobjects32;
//            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects32);

//            faceproposals.insert(faceproposals.end(), faceobjects32.begin(), faceobjects32.end());
//        }

//        // stride 16
//        {
//            ncnn::Mat score_blob, bbox_blob, landmark_blob;

//            const int base_size = 16;
//            const int feat_stride = 16;
//            ncnn::Mat ratios(1);
//            ratios[0] = 1.f;
//            ncnn::Mat scales(2);
//            scales[0] = 8.f;
//            scales[1] = 4.f;
//            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

//            std::vector<objBox> faceobjects16;
//            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects16);

//            faceproposals.insert(faceproposals.end(), faceobjects16.begin(), faceobjects16.end());
//        }

//        // stride 8
//        {
//            ncnn::Mat score_blob, bbox_blob, landmark_blob;

//            const int base_size = 16;
//            const int feat_stride = 8;
//            ncnn::Mat ratios(1);
//            ratios[0] = 1.f;
//            ncnn::Mat scales(2);
//            scales[0] = 2.f;
//            scales[1] = 1.f;
//            ncnn::Mat anchors = generate_anchors(base_size, ratios, scales);

//            std::vector<objBox> faceobjects8;
//            generate_proposals(anchors, feat_stride, score_blob, bbox_blob, landmark_blob, prob_threshold, faceobjects8);

//            faceproposals.insert(faceproposals.end(), faceobjects8.begin(), faceobjects8.end());
//        }

//        // sort all proposals by score from highest to lowest
//        qsort_descent_inplace(faceproposals);

//        // apply nms with nms_threshold
//        std::vector<int> picked;
//        nms_sorted_bboxes(faceproposals, picked, nms_threshold);

//        int face_count = picked.size();
//        std::vector<objBox> faceobjects;
//        faceobjects.resize(face_count);
//        for (int i = 0; i < face_count; i++)
//        {
//            faceobjects[i] = faceproposals[picked[i]];

//            // clip to image size
//            float x0 = faceobjects[i].rect.x;
//            float y0 = faceobjects[i].rect.y;
//            float x1 = x0 + faceobjects[i].rect.width;
//            float y1 = y0 + faceobjects[i].rect.height;

//            x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
//            y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
//            x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
//            y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);

//            faceobjects[i].rect.x = x0;
//            faceobjects[i].rect.y = y0;
//            faceobjects[i].rect.width = x1 - x0;
//            faceobjects[i].rect.height = y1 - y0;
//        }
//        res.push_back(faceobjects);
//    }

}
int faceDetectretinaTrt::Detect(akdData *data,std::vector<std::vector<objBox>> &res)
{

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

    postprocess_cpu(data,outputsizes,res);

    double t4=cv::getTickCount();

    std::cout<<"--"<<picnum<<"--time:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms ";
    std::cout<<(t3-t2)*1000/cv::getTickFrequency()<<"ms ";
    std::cout<<(t4-t3)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
    return 0;
}

odService* facedetectinit_retinaface_trt(odinitconfig config)
{
    faceDetectretinaTrt * ch=new faceDetectretinaTrt(config.modelpath,config.gpu,config.threadnum,config.maxnetsize,config.maxbatchsize);
    if(ch->initsuccess<0)
    {
        return nullptr;
    }
    return ch;
}


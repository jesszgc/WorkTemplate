#include"akdFaceI.h"
#include"od/odService.h"
#include"fr/frService.h"
#include"opencv2/opencv.hpp"
float norm_face112x112[]={38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366,41.5493, 92.3655, 70.7299, 92.2041};
std::vector< float > TransResult(const std::vector< float > & feat_points, const float std_points_ori[10])
{
    //原始点根据std求变换矩阵
    double sum_x_ori = 0, sum_y_ori = 0;
    double sum_u_ori = 0, sum_v_ori = 0;
    double sum_xx_yy_ori = 0;
    double sum_ux_vy_ori = 0;
    double sum_vx__uy_ori = 0;
    for (int c = 0; c < 5; ++c)
    {
        int x_off = c * 2;
        int y_off = x_off + 1;
        sum_x_ori += std_points_ori[c * 2];
        sum_y_ori += std_points_ori[c * 2 + 1];
        sum_u_ori += feat_points[x_off];
        sum_v_ori += feat_points[y_off];
        sum_xx_yy_ori += std_points_ori[c * 2] * std_points_ori[c * 2] +
                std_points_ori[c * 2 + 1] * std_points_ori[c * 2 + 1];
        sum_ux_vy_ori += std_points_ori[c * 2] * feat_points[x_off] +
                std_points_ori[c * 2 + 1] * feat_points[y_off];
        sum_vx__uy_ori += feat_points[y_off] * std_points_ori[c * 2] -
                feat_points[x_off] * std_points_ori[c * 2 + 1];
    }

    double q = sum_u_ori - sum_x_ori * sum_ux_vy_ori / sum_xx_yy_ori
            + sum_y_ori * sum_vx__uy_ori / sum_xx_yy_ori;
    double p = sum_v_ori - sum_y_ori * sum_ux_vy_ori / sum_xx_yy_ori
            - sum_x_ori * sum_vx__uy_ori / sum_xx_yy_ori;
    double rr = 5 - (sum_x_ori * sum_x_ori + sum_y_ori * sum_y_ori) / sum_xx_yy_ori;
    double tform_a = (sum_ux_vy_ori - sum_x_ori * q / rr - sum_y_ori * p / rr) / sum_xx_yy_ori;
    double tform_b = (sum_vx__uy_ori + sum_y_ori * q / rr - sum_x_ori * p / rr) / sum_xx_yy_ori;
    double tform_c = q / rr;
    double tform_d = p / rr;

    float feat_points_result_ori[10];
    for (int i = 0; i < 5; i++)
    {
        feat_points_result_ori[i * 2] = (feat_points[i * 2] - tform_c) / tform_a + (tform_b / tform_a)*((tform_a*(feat_points[i * 2 + 1] - tform_d) - tform_b*(feat_points[i * 2] - tform_c)) / sqrt(tform_a*tform_a + tform_b*tform_b));
        feat_points_result_ori[i * 2 + 1] = ((tform_a*(feat_points[i * 2 + 1] - tform_d) - tform_b*(feat_points[i * 2] - tform_c)) / sqrt(tform_a*tform_a + tform_b*tform_b));
    }
    float dis = 0.0f;
    for (int i = 0; i < 5; i++)
    {
        dis += (float)(sqrt((feat_points_result_ori[2 * i] - std_points_ori[2 * i])*(feat_points_result_ori[2 * i] - std_points_ori[2 * i]) + (feat_points_result_ori[2 * i + 1] - std_points_ori[2 * i + 1])*(feat_points_result_ori[2 * i + 1] - std_points_ori[2 * i + 1])));
    }
    std::vector< float > result;
    result.push_back(dis);
    result.push_back(tform_a);
    result.push_back(tform_b);
    result.push_back(tform_c);
    result.push_back(tform_d);

    return result;
}
cv::Mat getNormalFace(int w, int h, void* data, std::vector<cvpoint> &points)
{
    unsigned char* srcdata = (unsigned char*)data;
    std::vector< float > feat_points; feat_points.clear();
    feat_points.push_back(points[0].x);
    feat_points.push_back(points[0].y);
    feat_points.push_back(points[1].x);
    feat_points.push_back(points[1].y);
    feat_points.push_back(points[2].x);
    feat_points.push_back(points[2].y);
    feat_points.push_back(points[3].x);
    feat_points.push_back(points[3].y);
    feat_points.push_back(points[4].x);
    feat_points.push_back(points[4].y);
    //std::vector< float > result_ori = TransResult_ori(feat_points);
    std::vector< float > result_ori;// = TransResult(feat_points, norm_face112x112);
    int  roih = 112;
    int roiw = 112;
    result_ori = TransResult(feat_points, norm_face112x112);

    cv::Mat mtNorFace = cv::Mat::zeros(roiw, roih, CV_8UC3);

    float tform_a, tform_b, tform_c, tform_d;

    tform_a = result_ori[1];
    tform_b = result_ori[2];
    tform_c = result_ori[3];
    tform_d = result_ori[4];


    for (int r = 0; r < mtNorFace.rows; ++r)
    {
        unsigned char* pDst = mtNorFace.data + r * mtNorFace.cols * 3;
        for (int c = 0; c < mtNorFace.cols; ++c)
        {
            // Get the source position of each point on the destination feature map.
            double src_c = tform_a * c - tform_b * r + tform_c;
            double src_r = tform_b * c + tform_a * r + tform_d;
            int nsr = floor(src_r);
            int nsc = floor(src_c);
            if (nsc >= 0 && nsr >= 0 && nsr + 1 < h && nsc + 1 < w)
            {
                const unsigned char* pSrc = srcdata + nsr*w * 3;  //src_image.ptr<unsigned char>(nsr);
                const unsigned char* pSrcn = srcdata + (nsr + 1)*w * 3;//src_image.ptr<uchar>(nsr + 1);

                double cof_c = src_c - nsc;
                double cof_r = src_r - nsr;

                pDst[3 * c] = (1 - cof_r)*(1 - cof_c)*pSrc[3 * nsc] + cof_r*(1 - cof_c)*pSrcn[3 * nsc] + (1 - cof_r)*cof_c*pSrc[3 * nsc + 3] + cof_c*cof_r*pSrcn[3 * nsc + 3];
                pDst[3 * c + 1] = (1 - cof_r)*(1 - cof_c)*pSrc[3 * nsc + 1] + cof_r*(1 - cof_c)*pSrcn[3 * nsc + 1] + (1 - cof_r)*cof_c*pSrc[3 * nsc + 3 + 1] + cof_c*cof_r*pSrcn[3 * nsc + 3 + 1];
                pDst[3 * c + 2] = (1 - cof_r)*(1 - cof_c)*pSrc[3 * nsc + 2] + cof_r*(1 - cof_c)*pSrcn[3 * nsc + 2] + (1 - cof_r)*cof_c*pSrc[3 * nsc + 3 + 2] + cof_c*cof_r*pSrcn[3 * nsc + 3 + 2];
            }
            else
            {
                pDst[3 * c] = 0;
                pDst[3 * c + 1] = 0;
                pDst[3 * c + 2] = 0;
            }
        }
    }

    return mtNorFace;

}



akdFaceI::akdFaceI(faceiconfig config){
    odinitconfig odconfig;
    odconfig.modelpath=config.modelpath;

    odService* od=facedetectinit_retinaface_ncnn(odconfig);
    if(od!=nullptr)
    {
        fd=od;
        initsuccess=1;
    }
}
akdFaceI::~akdFaceI(){
    if(fd!=nullptr){
        odServiceRelease((odService*)fd);}
}
int akdFaceI::DetectFace(void* dataptr,int w,int h,int c,std::vector<faceBox> &res){

    akdData data;
    data.channels.push_back(c);
    data.heights.push_back(h);
    data.widths.push_back(w);
    data.ptr.push_back(dataptr);
    data.num=1;
    odService* od=(odService*)fd;
    std::vector<std::vector<objBox>> obs;
    od->Detect(&data,obs);
    for(int i=0;i<obs[0].size();++i)
    {
        objBox ob=obs[0][i];
        faceBox tmp;
        tmp.prob=ob.prob;
        tmp.rect=ob.rect;
        tmp.pp=ob.pp;
        res.push_back(tmp);

    }
    data.clean();
    return 0;
}



akdFaceFeatureI::akdFaceFeatureI(facefeatureiconfig config){
frinitconfig frconfig;
frconfig.modelpath=config.modelpath;
frconfig.maxbatchsize=config.maxbatchsize;
frconfig.gpu=config.gpu;
frconfig.threadnum=config.threadnum;
//frService* fr=frinit_arcface_ncnn(frconfig);
frService* fr=frinit_arcface_trt(frconfig);
if(fr!=nullptr)
{
    exer=fr;
    initsuccess=1;
}

}
akdFaceFeatureI::~akdFaceFeatureI(){

}
int akdFaceFeatureI::ExtractFeature(void* dataptr,int w,int h,int c,std::vector<float> &res)
{return -1;}
int akdFaceFeatureI::ExtractFeature(void* dataptr,int w,int h,int c,std::vector<faceBox> box,std::vector<std::vector<float>> &res)
{
    frService* fr=(frService*) exer;
    akdData data;
    data.num=0;
    std::vector<cv::Mat> faces;
    for(int i=0;i<box.size();++i){
        cv::Mat face=getNormalFace(w,h,dataptr,box[i].pp);

        faces.push_back(face);

        data.channels.push_back(face.channels());
        data.widths.push_back(face.cols);
        data.heights.push_back(face.rows);
        data.ptr.push_back(face.data);
        data.num++;
    }
    fr->Extract(&data,res);
    data.clean();
    return 0;
}
int akdFaceFeatureI::AddBase(std::string basename,std::vector<std::vector<float>> &basefeatures)
{
    this->base[basename]=basefeatures;
    return -1;
}
int akdFaceFeatureI::RmBase(std::string basename)
{
    return -1;
}
float similarity(std::vector<float> &f1,std::vector<float> &f2)
{
    float s=0;
    for(int i=0;i<f1.size();++i)
    {
        s+=(f1[i]*f2[i]);
    }
    return s;
}
int akdFaceFeatureI::Search(std::vector<float> &f,float &sim,std::string basename){
    std::vector<std::vector<float>> basefeatures=base[basename];

    int index=-1;
    sim=-1;
    for(int i=0;i<basefeatures.size();++i)
    {
        float stmp=similarity(f,basefeatures[i]);
        if(stmp>sim)
        {
            index=i;
            sim=stmp;
        }
    }
    return index;
}

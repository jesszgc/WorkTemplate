#ifndef AKDFACEI_H
#define AKDFACEI_H
#include"AkdComm.h"
#include<map>
struct faceBox{

    cvrect rect;
    int label;
    std::string labelname;
    float prob;
    std::vector<cvpoint> pp;
    
};

struct faceiconfig{
    std::string modelpath;
    int gpu;
    int threadnum;
    int maxbatchsize;
    int maxnetsize;
};

class EXPORT akdFaceI
{
    public:
    EXPORT akdFaceI(faceiconfig config);
     ~akdFaceI();
   EXPORT int DetectFace(void* dataptr,int w,int h,int c,std::vector<faceBox> &res);
    int initsuccess=-1;
   //int Detect(akdData *data,std::vector<std::vector<objBox>> &res)=0;
private:
    void*fd=nullptr;
};

struct facefeatureiconfig{
    std::string modelpath;
    int gpu;
    int threadnum;
    int maxbatchsize;
    int maxnetsize;
};

class EXPORT akdFaceFeatureI
{
public:
   EXPORT akdFaceFeatureI(facefeatureiconfig config);
    ~akdFaceFeatureI();
    int ExtractFeature(void* dataptr,int w,int h,int c,std::vector<float> &res);
    EXPORT int ExtractFeature(void* dataptr,int w,int h,int c,std::vector<faceBox> box,std::vector<std::vector<float>> &res);
    EXPORT int AddBase(std::string basename,std::vector<std::vector<float>> &basefeatures);
   EXPORT int RmBase(std::string basename);
   EXPORT int Search(std::vector<float> &f,float &sim,std::string basename);
public:
    int initsuccess=-1;
private:
    void* exer=nullptr;
    std::map<std::string,std::vector<std::vector<float>>> base;
};





//EXPORT odService* bodydetectinit_yolov5_ncnn(odinitconfig config);

//EXPORT int odServiceRelease(odService* od);

#endif

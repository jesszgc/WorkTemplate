#ifndef FRARCFACETRT_H
#define FRARCFACETRT_H
#include"AkdComm.h"
#include"../frService.h"

class frArcfaceNcnn:public frService
{
    public:
     frArcfaceNcnn(std::string modelpath);
     ~frArcfaceNcnn();
     int Extract(akdData *data,std::vector<std::vector<float>> &res);
public:
     int initsuccess=-1;
private:
     void* net;
     const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
     const float norm_vals[3] = {1.0 / 255.0, 1.0 /255.0, 1.0 / 255.0};
};




#endif

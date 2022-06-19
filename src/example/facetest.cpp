#include <opencv2/opencv.hpp>
#include <float.h>
#include <stdio.h>
#include <vector>
#include <deque>
#include<iostream>
#include<thread>
#include "../src/interface/akdfacei/akdFaceI.h"
#include<fstream>

int capthread(std::string path,cv::Mat &im,int &run,std::mutex &lock)
{
    cv::VideoCapture cap(path);
    while (run>0) {
        cv::Mat frame;
        if(false==cap.read(frame))
        {
            break;
        }
        lock.lock();
        im=frame.clone();
        lock.unlock();

    }
    run=-1;
}
int buildbase(akdFaceI *fd,akdFaceFeatureI* fr,std::string path,std::vector<std::string> &filenames,std::vector<std::vector<float>> &base)
{
    std::vector<std::string> files;
    cv::glob(path,files);
    for(int i=0;i<files.size();++i)
    {
        cv::Mat frame=cv::imread(files[i]);
        if(frame.empty()==true)
        {
            continue;
        }
        //cv::imshow("ff",frame);
       // cv::waitKey(0);
        std::vector<faceBox> boxs;
        fd->DetectFace(frame.data,frame.cols,frame.rows,frame.channels(),boxs);
        if(boxs.size()>0)
        {
            std::vector<std::vector<float>> ff;
            fr->ExtractFeature(frame.data,frame.cols,frame.rows,frame.channels(),boxs,ff);

            std::string name=files[i].substr(36,7);
            std::cout<<files[i]<<":"<<name<<std::endl;
            filenames.push_back(name);
            base.push_back(ff[0]);
        }
    }
    return 0;
}
struct faceinfo{
    cvrect loc;
    std::vector<cvpoint> pp;
    std::string name;
    float sim=-1;
};
int drawresult(cv::Mat &im,std::vector<faceinfo> box)
{
    for(int i=0;i<box.size();++i)
    {
        cv::rectangle(im,cv::Point(box[i].loc.x,box[i].loc.y),cv::Point(box[i].loc.x+box[i].loc.width,box[i].loc.y+box[i].loc.height),cv::Scalar(0,255,0),2);
        for(int j=0;j<box[i].pp.size();++j)
        {
            cv::circle(im,cv::Point(box[i].pp[j].x,box[i].pp[j].y),2,cv::Scalar(0,0,255),-1);
        }
        if(box[i].sim>0){
            std::string txt=std::to_string(box[i].sim)+" : "+box[i].name;
            cv::putText(im,txt,cv::Point(box[i].loc.x,box[i].loc.y),1,1.5,cv::Scalar(0,255,0));
        }
    }
    cv::namedWindow("res",cv::WINDOW_FULLSCREEN);
    cv::imshow("res",im);
    cv::waitKey(1);
    return 0;
}
int face_demo_ncnn()
{
    faceiconfig fdconfig;
    fdconfig.modelpath="/home/jesswork/coding/AkdCom/models/facedetect/mnet.25-opt";
    akdFaceI *fd=new akdFaceI (fdconfig);
    facefeatureiconfig frconfig;
    frconfig.modelpath="/home/jesswork/coding/AkdCom/models/facefeature/model-symbol-opt";
    akdFaceFeatureI* fr=new akdFaceFeatureI(frconfig);


    std::string basepicpath="/home/jesswork/coding/data/facebase";
    std::vector<std::string> basepersonnames;
    std::vector<std::vector<float>> base;
    buildbase(fd,fr,basepicpath,basepersonnames,base);

    fr->AddBase("default",base);

    int run=1;
    std::mutex lock;
//    std::string cappath="rtsp://admin:winter@2021@192.168.1.122:554/Streaming/Channels/101";
    std::string cappath="/home/work/data/face/test.mp4";
    cv::Mat im;
    std::thread cap(capthread,cappath,std::ref(im),std::ref(run),std::ref(lock));
    cap.detach();
    float threshold=0.1;
    while (run>0) {
        cv::Mat frame,frameshow;

        if(im.empty()==true||im.rows==0||im.cols==0)
        {
            continue;
        }
        lock.lock();
        frame=im.clone();
        lock.unlock();
        frameshow=frame.clone();
        double t1=cv::getTickCount();
        std::vector<faceBox> boxs;
        fd->DetectFace(frame.data,frame.cols,frame.rows,frame.channels(),boxs);

        double t2=cv::getTickCount();
        std::vector<faceinfo> res;
        std::vector<std::vector<float>> ffs;
        fr->ExtractFeature(frame.data,frame.cols,frame.rows,frame.channels(),boxs,ffs);

        for(int i=0;i<boxs.size();++i)
        {
            float sim;
            int index=fr->Search(ffs[i],sim,"default");

            faceinfo fi;
            fi.loc=boxs[i].rect;
            fi.pp=boxs[i].pp;
          if(index>=0)
           {
                fi.sim=sim;
                fi.name=basepersonnames[index];
            }
            res.push_back(fi);
        }


    double t3=cv::getTickCount();
    std::cout<<"fd fr:"<<(t2-t1)*1000/cv::getTickFrequency()<<" "<<(t3-t2)*1000/cv::getTickFrequency()<<std::endl;
    drawresult(frameshow,res);

}
run=-1;
delete fd;
delete fr;
return 0;
}

int face_demo_ncnn_single()
{
    faceiconfig fdconfig;
    fdconfig.modelpath="/home/jesswork/coding/AkdCom/models/facedetect/mnet.25-opt";
    akdFaceI *fd=new akdFaceI (fdconfig);
    facefeatureiconfig frconfig;
//    frconfig.modelpath="/home/jesswork/coding/AkdCom/models/facefeature/model-symbol-opt";
    frconfig.modelpath="/home/jesswork/coding/AkdCom/models/facefeature/model-r100-ii/model-0000.onnx";
    frconfig.maxbatchsize=20;
    akdFaceFeatureI* fr=new akdFaceFeatureI(frconfig);


    std::string basepicpath="/home/jesswork/coding/data/facebase";
    std::vector<std::string> basepersonnames;
    std::vector<std::vector<float>> base;
    buildbase(fd,fr,basepicpath,basepersonnames,base);

    fr->AddBase("default",base);

//    std::string cappath="rtsp://admin:winter@2021@192.168.1.122:554/Streaming/Channels/101";
    std::string cappath="/home/work/data/face/test.mp4";
   cv::VideoCapture cap(cappath);
    float threshold=0.1;
    while (27!=cv::waitKey(1)) {
        cv::Mat frame,frameshow;
        if(cap.read(frame)==false)
        {break;}


        frameshow=frame.clone();
        double t1=cv::getTickCount();
        std::vector<faceBox> boxs;
        fd->DetectFace(frame.data,frame.cols,frame.rows,frame.channels(),boxs);

        double t2=cv::getTickCount();
        std::vector<faceinfo> res;
        std::vector<std::vector<float>> ffs;
        fr->ExtractFeature(frame.data,frame.cols,frame.rows,frame.channels(),boxs,ffs);

        for(int i=0;i<boxs.size();++i)
        {
            float sim;
            int index=fr->Search(ffs[i],sim,"default");

            faceinfo fi;
            fi.loc=boxs[i].rect;
            fi.pp=boxs[i].pp;
          if(index>=0)
           {
                fi.sim=sim;
                fi.name=basepersonnames[index];
            }
            res.push_back(fi);
        }


    double t3=cv::getTickCount();
    std::cout<<"fd:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms      fr:"<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
    drawresult(frameshow,res);

}
delete fd;
delete fr;
return 0;
}


int main()
{
    face_demo_ncnn_single();
    //demo_trt();
    //demo_ncnn();
}



// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.


#include <opencv2/opencv.hpp>

#include <float.h>
#include <stdio.h>
#include <vector>
#include <deque>
#include<iostream>

#include "../src/od/odService.h"
#include "../src/lm/lmService.h"
#include<fstream>
static void draw_pose(const cv::Mat& bgr, inferOutdata &poses,std::vector<int> &pairs,int index)
{


    cv::Mat image = bgr;
    //cv::resize(image,image,cv::Size(640,640));
    for (int i = 0; i < poses.lms.size(); i++)
    {
        const landmark& obj = poses.lms[i];
        for(int j=0;j<obj.pp.size();++j)
        {

            cv::circle(image,cv::Point(obj.pp[j].x,obj.pp[j].y),4,cv::Scalar(0,0,255),-1);
            // cv::putText(image,cv::Point(obj.pp[j].x,obj.pp[j].y),4,cv::Scalar(0,0,255),-1);
            cv::putText(image, std::to_string(j), cv::Point(obj.pp[j].x,obj.pp[j].y+10),
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
        }
        for(int j=0;j<pairs.size();j+=2)
        {

            int index1=pairs[j];
            int index2=pairs[j+1];
            if(obj.pp[index1].prob<0.1||obj.pp[index2].prob<0.1)
            {
                continue;
            }
            cv::Point p1(obj.pp[index1].x,obj.pp[index1].y);
            cv::Point p2(obj.pp[index2].x,obj.pp[index2].y);
            cv::line(image,p1,p2,cv::Scalar(0,255,0),2,1);

        }

    }
    std::string showwindowname="showimage "+std::to_string(index);
    cv::imshow(showwindowname, image);
    cv::waitKey(10);
}

static void save_pose( inferOutdata poses,std::vector<int> &pairs,std::string savename)
{
    std::ofstream fileout(savename.c_str(),std::ios_base::out);
    int personnum=poses.lms.size();
    fileout<<std::to_string(personnum)<<std::endl;
    for(int i=0;i<personnum;++i)
    {
        landmark lm=poses.lms[i];
        for(int j=0;j<lm.pp.size();++j)
        {
            int x=lm.pp[j].x;
            int y=lm.pp[j].y;
            float prob=lm.pp[j].prob;
            fileout<<x<<std::endl;
            fileout<<y<<std::endl;
            fileout<<prob<<std::endl;
        }
    }
    fileout.close();
}


struct obj{
    std::deque<landmark> poses;
    int lost;
    int tagindex;
};

std::vector<cv::Scalar> color{cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(0,0,255),
            cv::Scalar(255,255,0),cv::Scalar(255,0,255),cv::Scalar(0,255,255),
            cv::Scalar(0,0,0),cv::Scalar(255,255,255),cv::Scalar(128,0,0),
            cv::Scalar(0,128,0),cv::Scalar(0,0,128),cv::Scalar(128,128,0),
            cv::Scalar(128,0,128),cv::Scalar(0,128,128),cv::Scalar(128,128,128),
            cv::Scalar(255,0,0),};

static int gettag(int capindex)
{
    static std::vector<int> tags(500,0);
    tags[capindex]++;
    return tags[capindex];
}

float iou(objBox ob1,objBox ob2)
{
    float overlap=(ob1.rect&ob2.rect).area();
    float res=overlap/(ob1.rect.area()+ob2.rect.area()-overlap);
    return res;
}
float posesimilarities(landmark &lm1,landmark &lm2)
{
    float overlap=(lm1.bbox&lm2.bbox).area();
    float res=overlap/(lm1.bbox.area()+lm2.bbox.area()-overlap);
    return res;
    //    std::vector<float> d(lm1.pp.size(),-1);
    //    float ds=0;
    //    int count=0;
    //    for(int i=0;i<lm1.pp.size();++i)
    //    {
    //        if(lm1.pp[i].prob>0.1&&lm2.pp[i].prob>0.1)
    //        {
    //            d[i]=(lm1.pp[i].x-lm2.pp[i].x)*(lm1.pp[i].x-lm2.pp[i].x)+(lm1.pp[i].y-lm2.pp[i].y)*(lm1.pp[i].y-lm2.pp[i].y);
    //            ds+=d[i];
    //            count++;
    //        }
    //    }

    //    ds/=count;

    //    return 1.0/ds;
}
int findbestmatchpose(landmark lm,std::vector<obj> &objs)
{
    int maxindex=-1;
    float maxiou=-9999;
    for(int i=0;i<objs.size();++i)
    {
        float ioutmp=posesimilarities(lm,objs[i].poses.back());
        //std::cout<<"iou:"<<ioutmp<<std::endl;
        if(ioutmp>maxiou
                &&ioutmp>0.3
                )
        {
            maxindex=i;
            maxiou=ioutmp;
        }
    }
    //std::cout<<"iou:"<<maxindex<<"  "<<maxiou<<std::endl;
    return maxindex;
}

int updateordelete(std::vector<obj> &obs,inferOutdata &boxs,int capindex )
{
    //delete
    for(auto it = boxs.lms.begin(); it != boxs.lms.end(); it++){

        //        if(it->label>0){
        //            //cout << "num is " << *it << endl;
        //            it = boxs.erase(it);
        //            if(it == boxs.end()) break;
        //        }
    }

    std::vector<int> matchindexs(boxs.lms.size());

    for(int i=0;i<boxs.lms.size();++i)
    {

        matchindexs[i]=findbestmatchpose(boxs.lms[i],obs);
    }
    std::cout<<"iou:"<<std::endl;
    std::vector<obj> obstmp;
    for(int i=0;i<matchindexs.size();++i)
    {
        if(matchindexs[i]>=0)
        {
            //update
            obs[matchindexs[i]].poses.push_back(boxs.lms[i]);
            if(obs[matchindexs[i]].poses.size()>20)
            {
                obs[matchindexs[i]].poses.pop_front();
            }
            obs[matchindexs[i]].lost=0;
        }else
        {

            // create
            obj objnew;
            objnew.poses.push_back(boxs.lms[i]);
            objnew.lost=-1;
            objnew.tagindex=gettag(capindex);

            obstmp.push_back(objnew);
        }
    }
    for(int i=0;i<obstmp.size();++i)
    {
        obs.push_back(obstmp[i]);
    }


    //delete
    for(auto it = obs.begin(); it != obs.end(); it++){
        it->lost++;
        if(it->lost > 9){
            //cout << "num is " << *it << endl;
            it = obs.erase(it);
            if(it == obs.end()) break;
        }



    }
    return 0;
}
int getbbox(landmark &obj,int &x,int &y,int &w,int &h)
{
    int minx=9999;
    int miny=9999;
    int maxx=-1;
    int maxy=-1;
    for(int i=0;i<obj.pp.size();++i)
    {
        if(obj.pp[i].prob>0.1)
        {
            if(minx>obj.pp[i].x)
            {minx=obj.pp[i].x;}
            if(miny>obj.pp[i].y)
            {miny=obj.pp[i].y;}

            if(maxx<obj.pp[i].x)
            {maxx=obj.pp[i].x;}
            if(maxy<obj.pp[i].y)
            {maxy=obj.pp[i].y;}
        }

    }
    x=minx;
    y=miny;
    w=maxx-minx;
    h=maxy-miny;
    return 0;
}
static void draw_pose_track(const cv::Mat& bgr, std::vector<obj> &obs,std::vector<int> &pairs,int index)
{


    cv::Mat image = bgr;
    //cv::resize(image,image,cv::Size(640,640));
    for (int i = 0; i < obs.size(); i++)
    {
        landmark obj =obs[i].poses.back();// poses.lms[i];


        cv::putText(image, std::to_string(obs[i].tagindex), cv::Point(obj.bbox.x,obj.bbox.y),
                    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0));
        cv::rectangle(image,cv::Rect(obj.bbox.x,obj.bbox.y,obj.bbox.width,obj.bbox.height),cv::Scalar(0,0,0));
        for(int j=0;j<obj.pp.size();++j)
        {

            cv::circle(image,cv::Point(obj.pp[j].x,obj.pp[j].y),4,cv::Scalar(0,0,255),-1);
            // cv::putText(image,cv::Point(obj.pp[j].x,obj.pp[j].y),4,cv::Scalar(0,0,255),-1);
            //            cv::putText(image, std::to_string(j), cv::Point(obj.pp[j].x,obj.pp[j].y+10),
            //                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
        }
        for(int j=0;j<pairs.size();j+=2)
        {

            int index1=pairs[j];
            int index2=pairs[j+1];
            if(obj.pp[index1].prob<0.1||obj.pp[index2].prob<0.1)
            {
                continue;
            }
            cv::Point p1(obj.pp[index1].x,obj.pp[index1].y);
            cv::Point p2(obj.pp[index2].x,obj.pp[index2].y);
            cv::line(image,p1,p2,cv::Scalar(0,255,0),2,1);

        }
        landmark obj1 =obs[i].poses[0];
        //getbbox(obj1,x,y,w,h);
        for(int j=1;j<obs[i].poses.size();j++)
        {
            landmark obj2 =obs[i].poses[j];
            //
            cv::Point p1(obj.bbox.x+obj.bbox.width/2,obj.bbox.y+obj.bbox.height/2);
            cv::Point p2(obj2.bbox.x+obj2.bbox.width/2,obj2.bbox.y+obj2.bbox.height/2);
            cv::line(image,p1,p2,color[obs[i].tagindex],1,1);
            obj =obj2;
        }
    }
    std::string showwindowname="showimage "+std::to_string(index);
    cv::imshow(showwindowname, image);
    cv::waitKey(10);
}

int demo_trt()
{
    lminitconfig config;
    config.modelpath="/home/jesswork/coding/AkdCom/models/openpose/openpose-sim_noprelu-sim_d.onnx";
    //config.modelpath="/home/jesswork/coding/AkdCom/models/openpose/pose_deploy-opt";
    config.gpu=0;
    config.threadnum=2;
    config.maxbatchsize=10;
    config.maxnetsize=640;


    int num=1;
    int showindex=0;
    lmService* lm=poseinit_openpose_trt(config);
    std::vector<std::vector<obj>> q_objs(num);
    if(lm==nullptr)
    {
        std::cout<<"init fail"<<std::endl;
        return -1;
    }
    akdData *data=new akdData();
    //    cv::VideoCapture cap("./sh.flv");
    cv::VideoCapture cap1("/home/work/data/bweb/hhrc.mp4");
    //cv::VideoCapture cap("/home/jess/data/bweb/sh.flv");
    //    cv::VideoCapture cap("/home/jess/data/bweb/北京冬奥会花样滑冰表演滑.flv");
    std::vector<int> pairs;
    lm->GetPairs(pairs);
    int w=cap1.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
    int h=cap1.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
    cv::VideoWriter ww("out.mp4",cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),25.0,cv::Size(w,h));
    while(27!=cv::waitKey(1))
    {
        //cv::Mat frame,frameshow;
        std::vector<cv::Mat> frames(num);
        std::vector<cv::Mat> frameshows(num);
        for(int i=0;i<num;++i){
            if(false==cap1.read(frames[i]))
            {
                break;
            }
            frameshows[i]=frames[i].clone();
        }
        //frame=cv::imread("/home/jess/data/bweb/飞书20220429-143402.jpg");
        //frameshow=frame.clone();
        for(int i=0;i<num;++i){
            data->channels.push_back(frames[i].channels());
            data->heights.push_back(frames[i].rows);
            data->widths.push_back(frames[i].cols);
            data->ptr.push_back(frames[i].data);
        }


        data->num=data->channels.size();
        std::vector<inferOutdata> objs;
        std::vector<inferOption> options;
        inferOption opt;
        opt.inferw=368;
        opt.inferh=368;
        opt.wholeimage=-1;
        options.push_back(opt);
        double t1=cv::getTickCount();

        lm->Infer(data,options,objs);
        double t2=cv::getTickCount();
        //draw_pose(frameshows[showindex],objs[showindex],pairs,showindex);
        updateordelete(q_objs[showindex],objs[showindex],showindex);
        draw_pose_track(frameshows[showindex],q_objs[showindex],pairs,showindex);
        //save_pose(objs[showindex],pairs,"");
        ww.write(frameshows[showindex]);
        //updateordelete(q_objs[showindex],objs[showindex],showindex);
        //draw_object(frameshows[showindex],q_objs[showindex],showindex);
        //draw_objects(frameshows[showindex],objs[showindex],showindex);
        double t3=cv::getTickCount();
        // std::cout<<"Detect Time:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms    ";
        // std::cout<<"Draw Time:"<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
        data->clean();
    }
    ww.release();
    lmServiceRelease(lm);
    delete data;


}



int facecapthread(std::string cappath,cv::Mat &dataq,int &run,std::mutex &lock,int &status)
{
    cv::VideoCapture cap(cappath);
    long long index=0;
    while (run>0) {
        cv::Mat frame;
        if(cap.read(frame)==false)
        {
            break;
        }

        lock.lock();
        dataq=frame.clone();
        lock.unlock();
    }
    return 0;
}
int capthread(std::string cappath,std::deque<cv::Mat> &dataq,int &index,int &run,std::mutex &lock,int &status)
{
    cv::VideoCapture cap(cappath);
    //long long index=0;
    while (run>0) {
        cv::Mat frame;
        if(cap.read(frame)==false)
        {
            break;
        }
        if(dataq.size()>300)
        {
            dataq.pop_front();
            //            dataindexq.pop_front();
        }
        lock.lock();
        dataq.push_back(frame);
        index++;
        //        dataindexq.push_back(index++);
        lock.unlock();
    }
    return 0;
}

struct personinfo{
    std::string name;
    double entertime;
    // body property
    //
    //........
    //
    // reid feature
    //
    //........
    //
};
std::vector<personinfo> infos;
int faceprocessthread(std::deque<cv::Mat>& data,std::mutex &lock)
{

    //match faceid to reid and bodyproperty
    // faceid
    //
    while(data.size()>0)
    { cv::Mat frame;
        int frameindex=0;
        lock.lock();
        frame=data.back().clone();
        lock.unlock();

        // process frame to get personinfo

        if(false)
        {
            personinfo p;
            infos.push_back(p);
        }

    }
    return -1;
}

int processthread(std::vector<std::deque<cv::Mat>>& datas,std::vector<int>& indexs,std::vector<std::mutex> &locks)
{
    lmService* lm;
    akdData* data;
    while(true)
    { std::vector<cv::Mat> frames(datas.size());
        //std::vector<int> frameindexs(datas.size());
        for(int i=0;i<datas.size();++i)
        {
            locks[i].lock();
            frames[i]=(datas[i].back());//.clone();
            indexs[i]=0;
            locks[i].unlock();
        }

        // process frames to get trigger

        std::vector<int> trigger(frames.size(),-1);
        for(int i=0;i<frames.size();++i)
        {
            if(trigger[i]<0)
            {
                continue;
            }
            std::vector<cv::Mat> clip;
            locks[i].lock();
            int idx=indexs[i];
            int start=datas[i].size()-idx-120;
            int end=datas[i].size()-idx;
            for(int j=start;j<end;++j)
            {
                clip.push_back(datas[i][j]);
            }
            locks[i].unlock();
        }

    }
    return -1;
}

int postprocessthread()
{
    return -1;
}
#include<thread>
int rundemo()
{
    std::vector<std::string> cappath{"","","","",""};
    std::vector<int> run(cappath.size(),1);
    std::vector<int> status(cappath.size(),1);
    std::vector<std::mutex> locks;
    std::vector<std::deque<cv::Mat>> data(cappath.size());
    std::vector<std::deque<long long>> dataindex(cappath.size());
    for(int i=0;i<cappath.size();++i)
    {
        std::thread tt(cappath,data[i],dataindex[i],run[i],locks[i],status[i]);
        tt.detach();
    }






}

int main()
{
    demo_trt();
    //demo_ncnn();
}


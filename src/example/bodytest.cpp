
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
static void draw_objects(const cv::Mat& bgr, const std::vector<objBox>& objects,int index)
{


    cv::Mat image = bgr.clone();
    cv::resize(image,image,cv::Size(640,640));
    for (size_t i = 0; i < objects.size(); i++)
    {
        const objBox& obj = objects[i];

        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //      obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        //
        //        if(obj.label>0)
        //        {
        //            continue;
        //        }
        cv::rectangle(image, cv::Rect(obj.rect.x,obj.rect.y,obj.rect.width,obj.rect.height), cv::Scalar(255, 0, 0));

        char text[256];
        //        if(obj.label!=0)
        //        {
        //            continue;
        //        }
        sprintf(text, "%s %.1f%%", obj.labelname.c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    std::string showwindowname="showimage "+std::to_string(index);
    //cv::imshow(showwindowname, image);
    // cv::waitKey(1);
}

struct obj{
    std::deque<objBox> locs;
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
static void draw_object(const cv::Mat& bgr,std::vector<obj> &obs,int index)
{
    cv::Mat image = bgr.clone();
    cv::resize(image,image,cv::Size(640,640));
    for (size_t i = 0; i < obs.size(); i++)
    {
        const obj& objtmp = obs[i];

        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //      obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        //
        //        if(obj.label>0)
        //        {
        //            continue;
        //        }
        for(int j=0;j<objtmp.locs.size()-1;++j)
        {
            cv::Point p1(objtmp.locs[j].rect.x+objtmp.locs[j].rect.width/2,objtmp.locs[j].rect.y+objtmp.locs[j].rect.height/2);
            cv::Point p2(objtmp.locs[j+1].rect.x+objtmp.locs[j+1].rect.width/2,objtmp.locs[j+1].rect.y+objtmp.locs[j+1].rect.height/2);
            cv::line(image,p1,p2,color[objtmp.tagindex%color.size()]);
        }

        objBox obj=objtmp.locs.back();
        cv::rectangle(image, cv::Rect(obj.rect.x,obj.rect.y,obj.rect.width,obj.rect.height), cv::Scalar(255, 0, 0));

        char text[256];
        //        if(obj.label!=0)
        //        {
        //            continue;
        //        }
        sprintf(text, "%d             %.1f%%", objtmp.tagindex, obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

//        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
//                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    std::string showwindowname="showimage "+std::to_string(index);
    cv::namedWindow(showwindowname,0);
    cv::imshow(showwindowname,image);
}

float iou(objBox ob1,objBox ob2)
{
    float overlap=(ob1.rect&ob2.rect).area();
    float res=overlap/(ob1.rect.area()+ob2.rect.area()-overlap);
    return res;
}
int findbestmatch(objBox &box,std::vector<obj> &obs)
{
    int maxindex=-1;
    float maxiou=-9999;
    for(int i=0;i<obs.size();++i)
    {
        float ioutmp=iou(box,obs[i].locs.back());
        if(ioutmp>maxiou&&ioutmp>0.3)
        {
            maxindex=i;
            maxiou=ioutmp;


        }

    }
    return maxindex;
}

int updateordelete(std::vector<obj> &obs,std::vector<objBox> &boxs,int capindex )
{
    //delete
    for(auto it = boxs.begin(); it != boxs.end(); it++){

        if(it->label>0){
            //cout << "num is " << *it << endl;
            it = boxs.erase(it);
            if(it == boxs.end()) break;
        }
    }

    std::vector<int> matchindexs(boxs.size());
    for(int i=0;i<boxs.size();++i)
    {

        matchindexs[i]=findbestmatch(boxs[i],obs);
    }
    std::vector<obj> obstmp;
    for(int i=0;i<matchindexs.size();++i)
    {
        if(matchindexs[i]>=0)
        {
            //update
            obs[matchindexs[i]].locs.push_back(boxs[i]);
            if(obs[matchindexs[i]].locs.size()>20)
            {
                obs[matchindexs[i]].locs.pop_front();
            }
            obs[matchindexs[i]].lost=0;
        }else
        {

            // create
            obj objnew;
            objnew.locs.push_back(boxs[i]);
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

int demo_trt()
{




    odinitconfig config;
    //config.modelpath="/home/jess/models/yolov5s.onnx";
    config.modelpath="/home/jesswork/coding/AkdCom/models/body/yolov5ssd.onnx";
//    config.modelpath="/home/jesswork/coding/AkdCom/models/openpose/openpose-sim_noprelu-sim.onnx";
    config.gpu=0;
    config.threadnum=2;
    config.maxbatchsize=20;
    config.maxnetsize=640;

    int num=2;
    int showindex=0;
    odService* od=bodydetectinit_yolov5_trt(config);
    std::vector<std::vector<obj>> q_objs(6);
    if(od==nullptr)
    {
        std::cout<<"init fail"<<std::endl;
        return -1;
    }
    akdData *data=new akdData();
    cv::VideoCapture cap("/home/work/data/bweb/s1.mp4");
    cv::VideoCapture cap1("/home/work/data/bweb/s2.mp4");
    //cv::VideoCapture cap("/home/jess/data/bweb/sh.flv");
    //    cv::VideoCapture cap("/home/jess/data/bweb/北京冬奥会花样滑冰表演滑.flv");

    while(27!=cv::waitKey(1))
    {
        //cv::Mat frame,frameshow;
        std::vector<cv::Mat> frames(num);
        std::vector<cv::Mat> frameshows(num);
        for(int i=0;i<num/2;++i){
            if(false==cap1.read(frames[i]))
            {
                break;
            }
            frameshows[i]=frames[i].clone();
        }
        for(int i=num/2;i<num;++i){
            if(false==cap.read(frames[i]))
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
        //cv::Mat img(data->heights[0],data->widths[0],CV_8UC3,data->ptr[0]);
        //cv::imshow("dsds",img);

        std::vector<std::vector<objBox>> objs;
        double t1=cv::getTickCount();
        od->Detect(data,objs);
        double t2=cv::getTickCount();
        updateordelete(q_objs[showindex],objs[showindex],showindex);
        draw_object(frameshows[showindex],q_objs[showindex],showindex);
        //draw_objects(frameshows[showindex],objs[showindex],showindex);
        double t3=cv::getTickCount();
        std::cout<<"Detect Time:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms    ";
        std::cout<<"Draw Time:"<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
        data->clean();
    }

    odServiceRelease(od);
    delete data;


}

int demo_ncnn()
{
    odinitconfig config;
    config.modelpath="/home/jess/work/ncnn-assets/models/yolov5s_6.0";
    config.gpu=0;
    config.threadnum=2;
    odService* od=bodydetectinit_yolov5_ncnn(config);
    if(od==nullptr)
    {
        std::cout<<"init fail"<<std::endl;
        return -1;
    }
    akdData *data=new akdData();
    cv::VideoCapture cap("/home/jess/data/bweb/【花样滑冰】中国双人历届冬奥、世锦赛、GPF合集3（PyeongChang周期） (P1. 2014 GPF SH SP).flv");
    //    cv::VideoCapture cap("/home/jess/data/bweb/北京冬奥会花样滑冰表演滑.flv");

    while(27!=cv::waitKey(1))
    {
        cv::Mat frame,frameshow;
        if(false==cap.read(frame))
        {
            break;
        }
        frameshow=frame.clone();
        data->channels.push_back(frame.channels());
        data->heights.push_back(frame.rows);
        data->widths.push_back(frame.cols);
        data->ptr.push_back(frame.data);
        data->num=1;
        std::vector<std::vector<objBox>> objs;
        double t1=cv::getTickCount();
        od->Detect(data,objs);
        double t2=cv::getTickCount();
        draw_objects(frameshow,objs[0],0);
        double t3=cv::getTickCount();
        std::cout<<"Detect Time:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms    ";
        std::cout<<"Draw Time:"<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
        data->clean();
    }




}

int main()
{
    //demo_ncnn();
    demo_trt();
}


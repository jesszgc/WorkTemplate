
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
    int startindex;
    int endindex;
    std::string starttag;
    std::string endtag;
};

std::vector<cv::Scalar> color{cv::Scalar(255,0,0),cv::Scalar(0,255,0),cv::Scalar(0,0,255),
            cv::Scalar(255,255,0),cv::Scalar(255,0,255),cv::Scalar(0,255,255),
            cv::Scalar(0,0,0),cv::Scalar(255,255,255),cv::Scalar(128,0,0),
            cv::Scalar(0,128,0),cv::Scalar(0,0,128),cv::Scalar(128,128,0),
            cv::Scalar(128,0,128),cv::Scalar(0,128,128),cv::Scalar(128,128,128),
            cv::Scalar(255,0,0),};



static void draw_pose_track(const cv::Mat& bgr, std::vector<obj> &obs,std::vector<int> &pairs,int index)
{


    cv::Mat image = bgr;
    //cv::resize(image,image,cv::Size(640,640));
    for (int i = 0; i < obs.size(); i++)
    {
        if(obs[i].poses.size()<10)
        {
            continue;
        }
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

struct trackinfo{
    inferOutdata boxs;
    int frameindex;
    std::string frametag;
};

class trackerservice{
public:
    std::string saveroot="/home/work/data/bweb/pose_person/";
public:trackerservice()
    {}
    ~trackerservice()
    {}
public:
    int savepose(obj &ob){
        std::string savename=saveroot+std::to_string(ob.tagindex)+".txt";
        std::ofstream fileout(savename.c_str(),std::ios_base::out);

        int personnum=ob.poses.size();
        fileout<<std::to_string(personnum)<<std::endl;
        fileout<<std::to_string(ob.startindex)<<std::endl;
        fileout<<std::to_string(ob.endindex)<<std::endl;
        for(int i=0;i<personnum;++i)
        {
            landmark lm=ob.poses[i];
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
    int updateordelete(std::vector<obj> &obs,trackinfo &ti,int capindex)
    { //delete bad pose
        for(auto it = ti.boxs.lms.begin(); it != ti.boxs.lms.end(); it++){

            //        if(it->label>0){
            //            //cout << "num is " << *it << endl;
            //            it = boxs.erase(it);
            //            if(it == boxs.end()) break;
            //        }
        }

        std::vector<int> matchindexs(ti.boxs.lms.size());

        for(int i=0;i<ti.boxs.lms.size();++i)
        {
            matchindexs[i]=findbestmatchpose(ti.boxs.lms[i],obs);
        }
        //std::cout<<"iou:"<<std::endl;
        std::vector<obj> obstmp;
        for(int i=0;i<matchindexs.size();++i)
        {
            if(matchindexs[i]>=0)
            {
                //update
                obs[matchindexs[i]].poses.push_back(ti.boxs.lms[i]);
                if(obs[matchindexs[i]].poses.size()>400)
                {
                    obs[matchindexs[i]].poses.pop_front();
                }
                obs[matchindexs[i]].lost=0;
                obs[matchindexs[i]].endindex=ti.frameindex;
            }else
            {
                // create
                obj objnew;
                objnew.poses.push_back(ti.boxs.lms[i]);
                objnew.lost=-1;
                objnew.tagindex=gettag(capindex);
                objnew.startindex=ti.frameindex;
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
                if(it->poses.size()>50)
                {
                    savepose(*it);
                }
                it = obs.erase(it);
                if(it == obs.end()) break;
            }



        }
        return 0;
    }
private:
    float iou(objBox ob1,objBox ob2)
    {
        float overlap=(ob1.rect&ob2.rect).area();
        float res=overlap/(ob1.rect.area()+ob2.rect.area()-overlap);
        return res;
    }
    static int gettag(int capindex)
    {
        static std::vector<int> tags(500,0);
        tags[capindex]++;
        return tags[capindex];
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
            if(ioutmp>maxiou
                    &&ioutmp>0.3
                    )
            {
                maxindex=i;
                maxiou=ioutmp;
            }
        }
        return maxindex;
    }

};

class trackerservice_1cap{
public:
    std::string saveroot="/home/work/data/bweb/pose_person/";
public:trackerservice_1cap()
    {}
    ~trackerservice_1cap()
    {
        for(auto it = obs.begin(); it != obs.end(); it++){

            if(it->poses.size()>50)
            {
                savepose(*it);
            }
        }
    }
public:
    int savepose(obj &ob){
        std::string savename=saveroot+std::to_string(ob.tagindex)+".txt";
        std::ofstream fileout(savename.c_str(),std::ios_base::out);

        int personnum=ob.poses.size();
        fileout<<std::to_string(personnum)<<std::endl;
        fileout<<std::to_string(ob.startindex)<<std::endl;
        fileout<<std::to_string(ob.endindex)<<std::endl;
        for(int i=0;i<personnum;++i)
        {
            landmark lm=ob.poses[i];
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
    int isposebad(landmark &lm)
    {
        int bad=0;
        for(int i=0;i<lm.pp.size();++i)
        {
            if(lm.pp[i].prob<0.2)
            {bad++;}
        }
        if(bad>20)
            return 1;
        else
            return -1;
    }
    int updateordelete(trackinfo &ti)
    { //delete bad pose
        for(auto it = ti.boxs.lms.begin(); it != ti.boxs.lms.end(); it++){
            if(isposebad(*it)>0){
                //cout << "num is " << *it << endl;
                it = ti.boxs.lms.erase(it);
                if(it == ti.boxs.lms.end()) break;
            }
        }

        std::vector<int> matchindexs(ti.boxs.lms.size());

        for(int i=0;i<ti.boxs.lms.size();++i)
        {
            matchindexs[i]=findbestmatchpose(ti.boxs.lms[i],obs);
        }
        // std::cout<<"iou:"<<std::endl;
        std::vector<obj> obstmp;
        for(int i=0;i<matchindexs.size();++i)
        {
            if(matchindexs[i]>=0)
            {
                //update
                obs[matchindexs[i]].poses.push_back(ti.boxs.lms[i]);
                if(obs[matchindexs[i]].poses.size()>800)
                {
                    obs[matchindexs[i]].poses.pop_front();
                }
                obs[matchindexs[i]].lost=0;
                obs[matchindexs[i]].endindex=ti.frameindex;
            }else
            {
                // create
                obj objnew;
                objnew.poses.push_back(ti.boxs.lms[i]);
                objnew.lost=-1;
                objnew.tagindex=gettag();
                objnew.startindex=ti.frameindex;
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
                if(it->poses.size()>50)
                {
                    savepose(*it);
                }
                it = obs.erase(it);
                if(it == obs.end()) break;
            }



        }
        return 0;
    }
private:
    float iou(cvrect ob1,cvrect ob2)
    {
        float overlap=(ob1&ob2).area();
        float res=overlap/(ob1.area()+ob2.area()-overlap);
        return res;
    }
    static int gettag()
    {
        static int tags=0;
        tags++;
        return tags;
    }
    float keypointdistance(KeyPoint &p1,KeyPoint &p2)
    {
        float d2=(p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y);
        return sqrt(d2);
    }
    float pairDistance(KeyPoint &p11,KeyPoint &p12,KeyPoint &p21,KeyPoint &p22)
    {
        float pair_d_1=sqrt((p11.x-p12.x)*(p11.x-p12.x)+(p11.y-p12.y)*(p11.y-p12.y));
        float pair_d_2=sqrt((p21.x-p22.x)*(p21.x-p22.x)+(p21.y-p22.y)*(p21.y-p22.y));
        float maxpair=pair_d_1>pair_d_2?pair_d_1:pair_d_2;
        float minpair=pair_d_1<pair_d_2?pair_d_1:pair_d_2;
        if((minpair/maxpair)<0.4)
        {
            return -1;
        }
        float d2_1=sqrt((p11.x-p21.x)*(p11.x-p21.x)+(p11.y-p21.y)*(p11.y-p21.y));
        float d2_2=sqrt((p12.x-p22.x)*(p12.x-p22.x)+(p12.y-p22.y)*(p12.y-p22.y));
        if((d2_1/minpair)>2||((d2_2/minpair)>2))
        {
            return -1;
        }
        return 1;
        //return (sqrt(d2_1)+sqrt(d2_2))/2.0;
    }
    int checkpair(landmark &lm1,landmark &lm2)
    {int bad=0;
        int count=0;
        float probsgreshold=0.3;
        for(int i=0;i<pairs.size();i+=2)
        {
            if(lm1.pp[pairs[i]].prob>probsgreshold&&lm1.pp[pairs[i+1]].prob>probsgreshold
                    &&lm2.pp[pairs[i]].prob>probsgreshold&&lm2.pp[pairs[i+1]].prob>probsgreshold)
            {
                count++;
                int badtmp=pairDistance(lm1.pp[pairs[i]],lm1.pp[pairs[i+1]],lm2.pp[pairs[i]],lm2.pp[pairs[i+1]]);
                if(badtmp<0)
                {bad++;}
            }
        }
        if(count==0)
        {
            return -1;
        }
        float badratio=float(bad)/float(count);
        if(badratio>0.5)
        {
            return -1;
        }else{
            return 1;
        }
    }
    float posesimilarities(landmark &lm1,landmark &lm2)
    {
        float ioutmp=iou(lm1.bbox,lm2.bbox);
        if(checkpair(lm1,lm2)<0)
        {
            return -1;
        }
        else
        {
            return ioutmp;
        }
        //        std::vector<float> d(lm1.pp.size(),-1);
        //        float ds=0;
        //        int count=0;
        //        for(int i=0;i<lm1.pp.size();++i)
        //        {
        //            if(lm1.pp[i].prob>0.1&&lm2.pp[i].prob>0.1)
        //            {
        //                d[i]=keypointdistance(lm1.pp[i],lm2.pp[i]);
        //                //d[i]=(lm1.pp[i].x-lm2.pp[i].x)*(lm1.pp[i].x-lm2.pp[i].x)+(lm1.pp[i].y-lm2.pp[i].y)*(lm1.pp[i].y-lm2.pp[i].y);
        //                ds+=d[i];
        //                count++;
        //            }
        //        }

        //        ds/=count;

        //    return 1.0/ds;
    }
    int findbestmatchpose(landmark lm,std::vector<obj> &objs)
    {
        int maxindex=-1;
        float maxiou=-9999;
        for(int i=0;i<objs.size();++i)
        {
            float ioutmp=posesimilarities(lm,objs[i].poses.back());
            if(ioutmp>maxiou
                    &&ioutmp>0.3
                    )
            {
                maxindex=i;
                maxiou=ioutmp;
            }
        }
        return maxindex;
    }
public:
    std::vector<obj> obs;
    std::vector<int> pairs;
};

int demo_trt()
{
    std::string caproot="/home/work/data/bweb/sh";
    std::string cappath=caproot+".flv";
    trackerservice ts;
    ts.saveroot=caproot+"/";
    lminitconfig config;
    config.modelpath="/home/jesswork/coding/AkdCom/models/openpose/openpose-sim_prelu.onnx";
    config.gpu=0;
    config.threadnum=2;
    config.maxbatchsize=10;
    config.maxnetsize=896;


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
    std::vector<cv::VideoCapture> caps(num);
    //    for(int i=0;i<caps.size();++i)
    //    {
    //        caps[i].open("");
    //    }
    cv::VideoCapture cap1(cappath);
    //    cap1.set()
    std::vector<int> pairs;
    lm->GetPairs(pairs);
    //    int w=cap1.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH);
    //    int h=cap1.get(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT);
    //    cv::VideoWriter ww("out.mp4",cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),25.0,cv::Size(w,h));
    int index=0;
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
        opt.inferw=896;
        opt.inferh=504;
        opt.wholeimage=-1;
        options.push_back(opt);
        double t1=cv::getTickCount();

        lm->Infer(data,options,objs);
        double t2=cv::getTickCount();
        //        draw_pose(frameshows[showindex],objs[showindex],pairs,showindex);
        trackinfo ti;
        ti.boxs=objs[showindex];
        ti.frameindex=index++;
        ts.updateordelete(q_objs[showindex],ti,showindex);
        //        std::string savename="/home/work/data/bweb/lnr1a/"+std::to_string(index++)+".txt";
        //        save_pose(objs[showindex],pairs,savename);
        draw_pose_track(frameshows[showindex],q_objs[showindex],pairs,showindex);
        //        ww.write(frameshows[showindex]);
        double t3=cv::getTickCount();
        // std::cout<<"Detect Time:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms    ";
        // std::cout<<"Draw Time:"<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
        data->clean();
    }
    //    ww.release();
    lmServiceRelease(lm);
    delete data;


}
int demo_trt_mulcap()
{
    std::vector<std::string> caproots={
//        "/home/work/data/bweb/dd.flv" ,
//        "/home/work/data/bweb/hhrc.mp4",
//        "/home/work/data/bweb/lnr1a.mp4",
//        "/home/work/data/bweb/p1.flv",
//        "/home/work/data/bweb/p2.flv",
        "/home/work/data/bweb/s2.mp4",
//        "/home/work/data/bweb/hbrj.mp4"
    };
    std::vector<cv::VideoCapture> caps(caproots.size());
    std::vector<trackerservice_1cap> tss(caproots.size());
    for(int i=0;i<caproots.size();++i)
    {
        std::string cappath=caproots[i];
        caps[i].open(cappath);
        tss[i].saveroot=caproots[i].substr(0,caproots[i].size()-4)+"/";
        std::string cmd="mkdir "+tss[i].saveroot;
        system(cmd.c_str());
    }
    lminitconfig config;
    config.modelpath="/home/jesswork/coding/AkdCom/models/openpose/openpose-sim_prelu.onnx";
    config.gpu=0;
    config.threadnum=2;
    config.maxbatchsize=caproots.size();
    config.maxnetsize=1904;


    int num=caproots.size();
    int showindex=0;
    lmService* lm=poseinit_openpose_trt(config);
    //    std::vector<std::vector<obj>> q_objs(num);
    if(lm==nullptr)
    {
        std::cout<<"init fail"<<std::endl;
        return -1;
    }
    akdData *data=new akdData();
    std::vector<int> pairs;
    lm->GetPairs(pairs);
    for(int i=0;i<tss.size();++i)
    {
    tss[i].pairs=pairs;
    }
    int index=0;
    while(27!=cv::waitKey(1))
    {
        //cv::Mat frame,frameshow;
        std::vector<cv::Mat> frames;//(num);
        std::vector<cv::Mat> frameshows(num);
        std::vector<int> capindexs;
        int bad=0;
        for(int i=0;i<num;++i){
            cv::Mat f;
            if(false==caps[i].read(f))
            {
                bad++;
                continue;
                //break;
            }
            cv::resize(f,f,cv::Size(1920,1080));
            frames.push_back(f);
            capindexs.push_back(i);
            frameshows[i]=f.clone();
        }
        if(bad>=num)
        {
            break;
        }

        for(int i=0;i<frames.size();++i){
            data->channels.push_back(frames[i].channels());
            data->heights.push_back(frames[i].rows);
            data->widths.push_back(frames[i].cols);
            data->ptr.push_back(frames[i].data);
        }


        data->num=data->channels.size();
        std::vector<inferOutdata> objs;
        std::vector<inferOption> options;
        inferOption opt;
        opt.inferw=1904;
        opt.inferh=1072;
        opt.wholeimage=-1;
        options.push_back(opt);
        double t1=cv::getTickCount();

        lm->Infer(data,options,objs);
        double t2=cv::getTickCount();
        //        draw_pose(frameshows[showindex],objs[showindex],pairs,showindex);
        for(int i=0;i<capindexs.size();++i){
            trackinfo ti;
            ti.boxs=objs[i];
            ti.frameindex=index;
            tss[capindexs[i]].updateordelete(ti);
        }
        std::cout<<"index:"<<index++<<std::endl;
        //        std::string savename="/home/work/data/bweb/lnr1a/"+std::to_string(index++)+".txt";
        //        save_pose(objs[showindex],pairs,savename);
        draw_pose_track(frameshows[showindex], tss[0].obs,pairs,0);
        //        ww.write(frameshows[showindex]);
        double t3=cv::getTickCount();
        // std::cout<<"Detect Time:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms    ";
        // std::cout<<"Draw Time:"<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
        data->clean();
    }
    //    ww.release();
    lmServiceRelease(lm);
    delete data;


}

//int demo_ncnn()
//{
//    lminitconfig config;
//    config.modelpath="/home/jesswork/coding/AkdCom/models/openpose/openpose-sim_noprelu-sim-opt";
//    config.gpu=0;
//    config.threadnum=2;
//    config.maxbatchsize=6;
//    config.maxnetsize=368;


//    int num=3;
//    int showindex=0;
//    lmService* lm=poseinit_openpose_ncnn(config);
//    std::vector<std::vector<obj>> q_objs(6);
//    if(lm==nullptr)
//    {
//        std::cout<<"init fail"<<std::endl;
//        return -1;
//    }
//    akdData *data=new akdData();
//    cv::VideoCapture cap("./sh.flv");
//    cv::VideoCapture cap1("/home/work/data/bweb/p2.flv");
//    //cv::VideoCapture cap("/home/jess/data/bweb/sh.flv");
//    //    cv::VideoCapture cap("/home/jess/data/bweb/北京冬奥会花样滑冰表演滑.flv");
//    std::vector<int> pairs;
//    lm->GetPairs(pairs);
//    while(27!=cv::waitKey(1))
//    {
//        //cv::Mat frame,frameshow;
//        std::vector<cv::Mat> frames(num);
//        std::vector<cv::Mat> frameshows(num);
//        for(int i=0;i<num;++i){
//            if(false==cap1.read(frames[i]))
//            {
//                break;
//            }
//            frameshows[i]=frames[i].clone();
//        }
//        //frame=cv::imread("/home/jess/data/bweb/飞书20220429-143402.jpg");
//        //frameshow=frame.clone();
//        for(int i=0;i<num;++i){
//            data->channels.push_back(frames[i].channels());
//            data->heights.push_back(frames[i].rows);
//            data->widths.push_back(frames[i].cols);
//            data->ptr.push_back(frames[i].data);
//        }


//        data->num=data->channels.size();
//        //cv::Mat img(data->heights[0],data->widths[0],CV_8UC3,data->ptr[0]);
//        //cv::imshow("dsds",img);

//        std::vector<inferOutdata> objs;
//        std::vector<inferOption> options;
//        inferOption opt;
//        opt.inferw=368;
//        opt.inferh=368;
//        opt.wholeimage=-1;
//        options.push_back(opt);
//        double t1=cv::getTickCount();

//        lm->Infer(data,options,objs);
//        double t2=cv::getTickCount();
//        draw_pose(frameshows[showindex],objs[showindex],pairs,showindex);
//        //updateordelete(q_objs[showindex],objs[showindex],showindex);
//        //draw_object(frameshows[showindex],q_objs[showindex],showindex);
//        //draw_objects(frameshows[showindex],objs[showindex],showindex);
//        double t3=cv::getTickCount();
//        std::cout<<"Detect Time:"<<(t2-t1)*1000/cv::getTickFrequency()<<"ms    ";
//        std::cout<<"Draw Time:"<<(t3-t2)*1000/cv::getTickFrequency()<<"ms"<<std::endl;
//        data->clean();
//    }

//    lmServiceRelease(lm);
//    delete data;


//}

int main()
{
    demo_trt_mulcap();
    //demo_ncnn();
}


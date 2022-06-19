#ifndef COMMON_H_
#define COMMON_H_

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;

class Shape
{
public:
	Shape() :num_(0), channels_(0), height_(0), width_(0), no_(0) {}
	Shape(int num, int channels, int height, int width, int no=1) :
		num_(num), channels_(channels), height_(height), width_(width), no_(no) {}

	inline const int num() { return num_; }
	inline const int channels() { return channels_; }
	inline const int height() { return height_; }
	inline const int width() { return width_; }
	inline const int no() { return no_; }
	inline const int count() { return num_ * channels_ * height_ * width_ * no_; }

	inline void set_height(int height) { height_ = height; }
	inline void set_width(int width) { width_ = width; }
	inline void set_no(int no) { no_ = no; }

	void Reshape(int num, int channels, int height, int width)
	{
		num_ = num;
		channels_ = channels;
		height_ = height;
		width_ = width;
	}

private:
	int num_;
	int channels_;
	int height_;
	int width_;
	int no_; // 模型在每一个anchor的输出数量(4(xywh) + 1(obj) + num_classes)
};

struct BoxInfo
{
public:
	int x1;
	int y1;
	int x2;
	int y2;
	float class_conf;
	float score;
	int class_idx;

	BoxInfo() : x1(0), y1(0), x2(0), y2(0), class_conf(0), score(0), class_idx(-1) {}
	BoxInfo(int lx, int ly, int rx, int ry, float conf, float s, int idx)
		: x1(lx), y1(ly), x2(rx), y2(ry), class_conf(conf), score(s), class_idx(idx) {}
};

class ComposeMatLambda
{
public:
    using FuncionType = std::function<cv::Mat(const cv::Mat&)>;

    ComposeMatLambda() = default;
    ComposeMatLambda(const vector<FuncionType>& lambda) :lambda_(lambda)
    {
        ;
    }
    cv::Mat operator()(cv::Mat& img)
    {
        for (auto func : lambda_)
            img = func(img);
        return img;
    }
private:
    vector<FuncionType> lambda_;
};

class LetterResize
{
public:
    LetterResize(cv::Size new_shape=cv::Size(640, 640),
        cv::Scalar color = cv::Scalar(114, 114, 114),
        int stride = 32) :new_shape_(new_shape), color_(color), stride_(stride) {}

    cv::Mat operator()(const cv::Mat& img)
    {
        int img_h = img.rows;
        int img_w = img.cols;

        int shape_h = new_shape_.height;
        int shape_w = new_shape_.width;
        cv::Mat resize_mat;
        cv::resize(img,resize_mat,cv::Size(shape_h,shape_w));
        return std::move(resize_mat);

    }

private:
    cv::Size new_shape_;
    cv::Scalar color_;
    int stride_;
};

class MatDivConstant
{
public:
    MatDivConstant(float constant) :constant_(constant) {}
    cv::Mat operator()(const cv::Mat& img)
    {
        cv::Mat tmp;
        cv::cvtColor(img, tmp, cv::COLOR_BGR2RGB);
        tmp.convertTo(tmp, CV_32FC3, 1, 0);
        tmp = tmp / constant_;
        return move(tmp);
    }

private:
    float constant_;
};

//void letter_resize(const cv::Mat& img, float& r,
//                   int& top, int& bottom, int& left, int& right,
//                   int stride = 32, cv::Size new_shape=cv::Size(640, 640))
//{

//    int img_h = img.rows;
//    int img_w = img.cols;

//    int shape_h = new_shape.height;
//    int shape_w = new_shape.width;

//    r = std::min(float(shape_h) / float(img_h), float(shape_w) / float(img_w));
//    cout << "r: " << r << endl;

//    cv::Size new_unpad = cv::Size(int(round(r * img_w)), int(round(r * img_h)));

//    int dw = new_shape.width - new_unpad.width;
//    int dh = new_shape.height - new_unpad.height;
//    dw = dw % stride;
//    dh = dh % stride;

//    float fdw = dw / 2.;
//    float fdh = dh / 2.;

//    top = int(round(fdh - 0.1));
//    bottom = int(round(fdh + 0.1));
//    left = int(round(fdw - 0.1));
//    right = int(round(fdw + 0.1));
//}

#endif

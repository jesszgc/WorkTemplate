
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

int demo()
{
    cv::Mat im = cv::imread("D:/BaiduNetdiskDownload/Result_t_23.jpg");
    cv::namedWindow("im", 0);
    cv::imshow("im", im);
    cv::waitKey(0);
    return -1;
}
int main()
{
    //demo_ncnn();
    demo();
}


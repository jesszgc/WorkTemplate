#pragma once
#ifndef _PersonPosePt
#define _PersonPosePt
struct PosePt
{
	float fx;
	float fy;
	float fs;
};
#endif // !_PersonPosePt

#ifndef _PersonPoseImg
#define _PersonPoseImg
struct PoseImg
{
	int w;
	int h;
	unsigned char* data;
};
#endif // !_PersonPoseImg

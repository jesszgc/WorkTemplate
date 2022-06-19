#ifndef YZCOMMON_H
#define YZCOMMON_H
#include <cstring>
#include <string>
#include <vector>
#include<math.h>
#include <algorithm>

#if defined(_MSC_VER) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#  define DECL_EXPORT __declspec(dllexport)
#  define DECL_IMPORT __declspec(dllimport)
#else
#  define DECL_EXPORT     __attribute__((visibility("default")))
#  define DECL_IMPORT     __attribute__((visibility("default")))
#endif

#if defined(libYz_EXPORTS)
#  define EXPORT DECL_EXPORT
#else
#  define EXPORT DECL_IMPORT
#endif


enum InferType {
	GPU_FP32 = 1,
	GPU_FP16,
	GPU_Int8,
	CPU
};

enum InDataType
{
	CPU_BGR = 0,
	CPU_RGB,
	GPU_CUDA_BGR,
	GPU_CUDA_RGB,
	GPU_CUDA_NV12,
	GPU_GpuMat,
	GPU_GpuMat_BGRA
	//GPU_RGBA,
	//GPU_YUVA,
	//GPU_NV12A,

};

typedef unsigned char BYTE;
namespace cv {
	class Mat;
	//class Rect;
}
typedef struct _cvsize
{
	int w;
	int h;
	_cvsize()
	{
		w = 0;
		h = 0;
	}
	_cvsize(int xx, int yy)
	{
		w = xx;
		h = yy;
	}
}cvsize;
typedef struct _cvpoint
{
	int x;
	int y;
	_cvpoint()
	{
		x = 0;
		y = 0;
	}
	_cvpoint(int xx, int yy)
	{
		x = xx;
		y = yy;
	}
}cvpoint;
typedef struct _cvrect
{
	int x;
	int y;
	int width;
	int height;

	_cvrect()
	{
		x = 0;
		y = 0;
		width = 0;
		height = 0;
	}
	_cvrect(int xx, int yy, int w, int h)
	{
		x = xx;
		y = yy;
		width = w;
		height = h;
	}
	//friend class cv::Rect;
	//_cvrect(cv::Rect rr)
	//{
	//	x = rr.x;
	//	y = rr.y;
	//	width = rr.Width;
	//	height =rr.Height;
	//}
	int area()
	{
		if (width < 0 || height < 0)
		{
			return 0;
		}
		else {
			return width * height;
		}
	}
	_cvrect operator &(_cvrect b)
	{
		int x1 = std::max(x, b.x);
		int y1 = std::max(y, b.y);
		int width1 = std::min(x + width, b.x + b.width) - x1;
		int height1 = std::min(y + height, b.y + b.height) - y1;
		cvrect a(x1, y1, width1, height1);
		if (width1 <= 0 || height1 <= 0)
			cvrect a(0, 0, -1, -1);
		return a;
	}
	bool contains(cvpoint pt)
	{
		return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
	}


}cvrect;

class cvmat
{
public:
	BYTE * pBuf=nullptr;
	int nWidth;
	int nHeight;
	int nChannel;
	void *dataptr=nullptr;
//public:
//	friend class cv::Mat;
public:
	EXPORT cvmat();
	EXPORT cvmat(const cvmat& cd);
	EXPORT cvmat(int w, int h, int c, const void* data);
	EXPORT cvmat& operator=(const cvmat& cd);
	EXPORT bool empty();
	EXPORT ~cvmat();
	
};
int cvmatresize(const cvmat& src, cvmat& dst, cvsize size);

namespace YzFace {
	class YzFaceI;
	class YzFeatureI;
	
};
class BodyDetectClassy;

struct YzInData
{
	std::string channel_context = "";
	std::string context = "";

	void* img=nullptr;
	int height=-1;
	int width = -1;
	int channles = -1;
	int step = -1;
	InDataType datatype=InDataType::CPU_BGR;
};

class YzData
{
	public:
		EXPORT YzData() {};
		EXPORT ~YzData();
		EXPORT int init(int w, int h, int c, int batchsize, int gpudevice);

		EXPORT int PutData(std::vector<YzInData> &data);

	private:
		void *im_bgrchar_gpu = nullptr;
		int w_gpu=-1;
		int h_gpu=-1;
		int batchsize=0;
		int gpu=0;

		std::vector< void* > im_bgrchar_cpus{};
		std::vector< int > w_cpus{};
		std::vector< int > h_cpus{};

		int picnum=0;
		int GpuOrCpu=0;

		friend class YzFace::YzFaceI;
		friend class YzFace::YzFeatureI;
		friend class BodyDetectClassy;
		friend class HeadDetectClassy;
		friend class HelmetDetectClassy;
		friend class YzCount;
};
struct ObjProperty
{
	int label;
	std::string lablename;
	float prob;
};

struct DetectClassyinitconfig {
	std::string od_modelpath;
	std::string classy_modelpath;
	int gpu = 0;
	int Height = 1080;
	int Width = 1920;
	int miniobj = 80;
	int maxobj = 500;

	int od_batchsize = 100;
	int classy_batchsize = 100;

	float threshold;

	InferType infertype = GPU_FP32;
	int classytype = 1;
};

struct DetectClassyoutdata {

	std::string channel_context;
	std::string context;

	// ob rect
	int x=0;
	int y=0;
	int height=-1;
	int width=-1;

	cvrect objrect;

	int label=-1;
	std::string name="";
	float prob=-1.0f;
	float prob_od=-1.0f;

	std::vector<ObjProperty> properties{};
};

#endif
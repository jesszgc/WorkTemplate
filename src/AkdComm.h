#ifndef AKDCOMM_H
#define AKDCOMM_H
#include<string>
#include<vector>
#if defined(_MSC_VER) || defined(WIN64) || defined(_WIN64) || defined(__WIN64__) || defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
#  define DECL_EXPORT __declspec(dllexport)
#  define DECL_IMPORT __declspec(dllimport)
#else
#  define DECL_EXPORT     __attribute__((visibility("default")))
#  define DECL_IMPORT     __attribute__((visibility("default")))
#endif

#if defined(libAkd_EXPORTS)
#  define EXPORT DECL_EXPORT
#else
#  define EXPORT DECL_IMPORT
#endif

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
		_cvrect a(x1, y1, width1, height1);
		if (width1 <= 0 || height1 <= 0)
			_cvrect a(0, 0, -1, -1);
		return a;
	}
	bool contains(cvpoint pt)
	{
		return x <= pt.x && pt.x < x + width && y <= pt.y && pt.y < y + height;
	}


}cvrect;


struct akdData{
    void* data=nullptr;
    int num;
    std::vector<void*> ptr;
    std::vector<int> widths;
    std::vector<int> heights;
    std::vector<int> channels;
    std::vector<int> step;
    std::vector<int> datatype;
//    int inferw;
//    int inferh;
    int clean()
    {
        ptr.clear();
        widths.clear();
        heights.clear();
        channels.clear();
        step.clear();
        datatype.clear();
        num=0;
    }
};
#endif

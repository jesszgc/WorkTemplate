#include"Comm.h"
#include<iostream>

cvmat::cvmat()
{
	pBuf = nullptr;
	dataptr = nullptr;
	nWidth = -1;
	nHeight = -1;
	nChannel = -1;
}
cvmat::cvmat(const cvmat& cd)
{
	this->pBuf = new BYTE[cd.nWidth * cd.nHeight * cd.nChannel];
	memcpy(this->pBuf, cd.pBuf, cd.nWidth * cd.nHeight * cd.nChannel * sizeof(BYTE));
	this->dataptr = this->pBuf;
	this->nChannel = cd.nChannel;
	this->nHeight = cd.nHeight;
	this->nWidth = cd.nWidth;
}
cvmat::cvmat(int w, int h, int c, const void* data)
{
	this->pBuf = new BYTE[w * h * c * sizeof(BYTE)];
	memcpy(this->pBuf, data, w * h * c * sizeof(BYTE));
	this->dataptr = this->pBuf;
	this->nChannel = c;
	this->nHeight = h;
	this->nWidth = w;
}

cvmat& cvmat::operator=(const cvmat& cd)
{
	if (this == &cd)
	{
		return *this;
	}
	this->~cvmat();
	this->pBuf = new BYTE[cd.nWidth * cd.nHeight * cd.nChannel];
	memcpy(this->pBuf, cd.pBuf, cd.nWidth * cd.nHeight * cd.nChannel * sizeof(BYTE));
	this->dataptr = this->pBuf;
	this->nChannel = cd.nChannel;
	this->nHeight = cd.nHeight;
	this->nWidth = cd.nWidth;
	return *this;

}
bool cvmat::empty()
{
	if (pBuf == nullptr)
	{
		return true;
	}
	else
	{
		return false;
	}
}
cvmat::~cvmat()
{
	if (pBuf != nullptr && nWidth != -1 && nHeight != -1 && nChannel != -1) {
		delete pBuf;
		pBuf = nullptr;
		dataptr = nullptr;
		nWidth = -1;
		nHeight = -1;
		nChannel = -1;
	}
}


int cvmatresize(const cvmat& src, cvmat& dst, cvsize size)
{
	return 0;
}





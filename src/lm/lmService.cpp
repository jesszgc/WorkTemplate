#include"lmService.h"

lmService::lmService()
{}
lmService::~lmService()
{}
int lmServiceRelease(lmService* od)
{
    if(od!=nullptr)
    {
        delete od;
        od=nullptr;
    }
    return -1;
}

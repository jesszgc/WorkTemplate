#include"odService.h"

odService::odService()
{}
odService::~odService()
{}
int odServiceRelease(odService* od)
{
    if(od!=nullptr)
    {
        delete od;
        od=nullptr;
    }
    return -1;
}

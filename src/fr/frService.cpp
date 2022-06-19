#include"frService.h"

frService::frService()
{}
frService::~frService()
{}
int frServiceRelease(frService* od)
{
    if(od!=nullptr)
    {
        delete od;
        od=nullptr;
    }
    return -1;
}

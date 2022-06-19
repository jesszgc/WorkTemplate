#include "cudaService.h"
namespace akdcuda {


cudaService::cudaService()
{

}
void* safeCudaMalloc(long memSize)
{
    void* deviceMem = nullptr;
    cudaError cudaerror = cudaMalloc(&deviceMem, memSize);
    if (deviceMem == nullptr || cudaerror != cudaSuccess)
    {
        std::cout << "cuda Out of memory" << std::endl;
        exit(-1);
    }
    return deviceMem;
}
}

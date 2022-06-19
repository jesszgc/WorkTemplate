/*
 * @Author: zerollzeng
 * @Date: 2019-10-10 18:07:54
 * @LastEditors: zerollzeng
 * @LastEditTime: 2019-10-16 09:07:41
 */
#ifndef OPENPOSE_GPU_CUDA_HPP
#define OPENPOSE_GPU_CUDA_HPP

#include <utility> // std::pair
#include <vector>
#include <array>
#include <iostream>

namespace op
{
    
    const auto CUDA_NUM_THREADS = 512u;


    inline unsigned int getNumberCudaBlocks(
        const unsigned int totalRequired, const unsigned int numberCudaThreads = CUDA_NUM_THREADS)
    {
        return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
    }

}

#endif // OPENPOSE_GPU_CUDA_HPP

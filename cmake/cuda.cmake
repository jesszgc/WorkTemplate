


if(${MSVC_TOOLSET_VERSION} MATCHES "142")
	set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4")
	set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4")
endif()
if(${MSVC_TOOLSET_VERSION} MATCHES "140")
	set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
	set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.2")
endif()

#cmake_policy(SET CMP0074 NEW)

find_package(CUDA REQUIRED)
    
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} /MD")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -rdynamic")
ADD_DEFINITIONS(-DCUDA_FP16)

include(cmake/FindcuBLAS.cmake)
find_package(cuBLAS)
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
SET(CUDA_NVCC_FLAGS
    -gencode arch=compute_50,code=sm_50;
   -gencode arch=compute_52,code=sm_52;
   -gencode arch=compute_60,code=sm_60;
   -gencode arch=compute_61,code=sm_61;
   -gencode arch=compute_70,code=sm_70;
   -gencode arch=compute_75,code=sm_75;
   -std=c++11;-O0;-G;-g)
endif()

if(${MSVC_TOOLSET_VERSION} MATCHES "142")
 SET(CUDA_NVCC_FLAGS 
     -gencode arch=compute_50,code=sm_50; 
    -gencode arch=compute_52,code=sm_52;
    -gencode arch=compute_60,code=sm_60; 
    -gencode arch=compute_61,code=sm_61; 
    -gencode arch=compute_70,code=sm_70; 
    -gencode arch=compute_75,code=sm_75;
    -std=c++11;-O3;-G;-g)
endif()


if(${MSVC_TOOLSET_VERSION} MATCHES "142")
 SET(CUDA_NVCC_FLAGS 
     -gencode arch=compute_50,code=sm_50; 
    -gencode arch=compute_52,code=sm_52;
    -gencode arch=compute_60,code=sm_60; 
    -gencode arch=compute_61,code=sm_61; 
    -gencode arch=compute_70,code=sm_70; 
    -gencode arch=compute_75,code=sm_75;
    -gencode arch=compute_80,code=sm_80;
    -gencode arch=compute_86,code=sm_86;
    -std=c++11;-O3;-G;-g)

endif()

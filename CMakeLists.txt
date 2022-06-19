# ENABLE_LANGUAGE(ASM)


cmake_minimum_required(VERSION 3.4.1)
project(YzFaceWork)
option(YzFaceWork_OPENMP "openmp support" ON)

option(YzFaceWork_CMAKE_VERBOSE "print verbose cmake messages" OFF)
option(YzFaceWork_VULKAN "vulkan compute support" OFF)
option(YzFaceWork_AVX2 "optimize x86 platform with avx2" OFF)
option(YzFaceWork_COVERAGE "build for coverage" OFF)

option(WITH_CUDA "build benchmark" ON)
option(FEISI "build feisi interface" ON)
option(YzFace "build yzFace interface" ON)
option(YzBody "build yzbody interface" ON)
option(YzHead "build yzbody interface" ON)
option(YzSafeHat "build yzbody interface" ON)
option(YzOCR "build yzbody interface" ON)
option(YzCount "build yzbody interface" ON)
option(YzOCR_paddle "build yzbody interface" OFF)
ADD_DEFINITIONS(-DlibYz_EXPORTS)



option(OPENCV "minimal opencv structure emulation" ON)

#include(cmake/utils.cmake)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

include(cmake/ncnn.cmake)
include_directories(${CMAKE_SOURCE_DIR}/src)


include("./yzbody.cmake")
include("./yzhead.cmake")
include("./yzheadcount.cmake")
include("./yzhelmet.cmake")
include("./yzface.cmake")
#include("./yzocr.cmake")
if(${MSVC_TOOLSET_VERSION} MATCHES "142")
include("./yzocr_paddle.cmake")
endif()
add_subdirectory(demo)
add_subdirectory(Tools)
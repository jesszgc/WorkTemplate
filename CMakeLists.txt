cmake_minimum_required(VERSION 2.8.12)

project(AkdWork)
option(AkdBODY "" ON)
option(AkdFACE "" ON)
ADD_DEFINITIONS(-DlibAkd_EXPORT)

SET(CMAKE_CXX_FLAGS "-std=c++11 -pthread ${CMAKE_CXX_FLAGS} ")
if(POLICY CMP0025)
    # reference from https://cmake.org/cmake/help/latest/policy/CMP0025.html
    cmake_policy(SET CMP0025 NEW)
endif()


find_package(OpenMP)  
if(OPENMP_FOUND)
message("OPENMP FOUND")
else()
message("OPENMP NOT FOUND")
endif()

include(cmake/ncnn.cmake)
include_directories(${CMAKE_SOURCE_DIR}/src)

include("./akdbody.cmake")
include("./akdface.cmake")

add_subdirectory(example)
add_subdirectory(app)

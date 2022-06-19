
## 判断编译器
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
# using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
# using GCC

set(NCNN_DIR "${CMAKE_SOURCE_DIR}/../3rd/ncnn/ncnn-20220420-ubuntu-1804")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
# using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
# using Visual Studio C++
	message(${MSVC_TOOLSET_VERSION})
	#if(${MSVC_TOOLSET_VERSION} MATCHES "142")
	
	#set(NCNN_DIR "${CMAKE_SOURCE_DIR}/../3rd/ncnn/ncnn_vulkan_20210720/vc16")
	#endif()

	#if(${MSVC_TOOLSET_VERSION} MATCHES "140")
	
	#set(NCNN_DIR "${CMAKE_SOURCE_DIR}/../3rd/ncnn/ncnn_vulkan_20210720/vc14")
	#endif()

endif()


include_directories("${NCNN_DIR}/include/")
include("${NCNN_DIR}/lib/cmake/OSDependentTargets.cmake")
include("${NCNN_DIR}/lib/cmake/OGLCompilerTargets.cmake")
include("${NCNN_DIR}/lib/cmake/glslangTargets.cmake")
include("${NCNN_DIR}/lib/cmake/SPIRVTargets.cmake")
include("${NCNN_DIR}/lib/cmake/ncnn/ncnnConfig.cmake")







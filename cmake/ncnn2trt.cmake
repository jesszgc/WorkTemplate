## 判断编译器
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
# using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
# using GCC
		set(trtcvtm_dir ${CMAKE_SOURCE_DIR}/../ncnn2trt/build/install)
		include_directories(${trtcvtm_dir}/include)
		link_directories(${trtcvtm_dir}/lib)
		set(trtcvtm_LIB ${trtcvtm_dir}/lib/libncnn2tensorrtLib.so)

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
# using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
# using Visual Studio C++



if(${MSVC_TOOLSET_VERSION} MATCHES "142")
		set(trtcvtm_dir ${CMAKE_SOURCE_DIR}/../3rd/ConvertModel_vs19)
		include_directories(${trtcvtm_dir}/include)
		link_directories(${trtcvtm_dir}/bin)
		set(trtcvtm_LIB ${trtcvtm_dir}/lib/*.lib)
		endif()

		if(${MSVC_TOOLSET_VERSION} MATCHES "140")
		set(trtcvtm_dir ${CMAKE_SOURCE_DIR}/../3rd/ConvertModelDll/ncnn2trt)
		include_directories(${trtcvtm_dir})
		link_directories(${trtcvtm_dir})
		set(trtcvtm_LIB ${trtcvtm_dir}/*.lib)
		endif()
endif()





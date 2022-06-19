		

## 判断编译器
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
# using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
# using GCC
                set(tensorrt_dir ${CMAKE_SOURCE_DIR}/../3rd/TensorRT-8.4.0.6)
#            set(tensorrt_dir ${CMAKE_SOURCE_DIR}/../3rd/TensorRT-8.2.0.6)
		
        include_directories(${tensorrt_dir}/include)
		include_directories(${tensorrt_dir}/samples/common)
                link_directories(${tensorrt_dir}/targets/x86_64-linux-gnu/lib)
                set(TRT_LIB ${tensorrt_dir}/targets/x86_64-linux-gnu/lib/libnvinfer.so)
                set(TRT_LIB ${TRT_LIB} ${tensorrt_dir}/targets/x86_64-linux-gnu/lib/libnvonnxparser.so)
                set(TRT_LIB ${TRT_LIB} ${tensorrt_dir}/targets/x86_64-linux-gnu/lib/libnvparsers.so)
                set(TRT_LIB ${TRT_LIB} ${tensorrt_dir}/targets/x86_64-linux-gnu/lib/libnvinfer_plugin.so)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
# using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
# using Visual Studio C++
		if(${MSVC_TOOLSET_VERSION} MATCHES "142")
		set(tensorrt_dir ${CMAKE_SOURCE_DIR}/../3rd/TensorRT/TensorRT-8.2.0.6.Windows10.x86_64.cuda-11.4.cudnn8.2)
		endif()
        if(${MSVC_TOOLSET_VERSION} MATCHES "140")
		set(tensorrt_dir ${CMAKE_SOURCE_DIR}/../3rd/TensorRT/TensorRT-8.0.1.6.Windows10.x86_64.cuda-10.2.cudnn8.2)
		endif()

        include_directories(${tensorrt_dir}/include)
		include_directories(${tensorrt_dir}/samples/common)
		link_directories(${tensorrt_dir}/lib)
		set(TRT_LIB ${tensorrt_dir}/lib/*.lib)
endif()





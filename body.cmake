if(AkdBODY)
# if(OPENCV)
#     ADD_DEFINITIONS(-DWITH_OpenCV)
#     # 设置opencv的静态库
#     # set(OpenCV_STATIC ON)
    # set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../3rd/cv343_world")
    # find_package(OpenCV REQUIRED)
    
# endif()
if(1)
    ADD_DEFINITIONS(-DWITH_CUDA)
    include(cmake/cuda.cmake)
    
    option(WITH_TRT "build benchmark" ON)
    if(WITH_TRT)
        ADD_DEFINITIONS(-DWITH_TRT)
        include(cmake/trt.cmake)
        #include(cmake/ncnn2trt.cmake)
    endif()
endif()
#
include(cmake/utils.cmake)
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../3rd/opencv/lib/cmake/opencv4")

find_package(OpenCV REQUIRED)
set(cvlib ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB native_srcs__tmp
    ${CMAKE_SOURCE_DIR}/src/*.cpp
    ${CMAKE_SOURCE_DIR}/src/*.h
    ${CMAKE_SOURCE_DIR}/src/3rdwrap/trt/*.cpp
    ${CMAKE_SOURCE_DIR}/src/3rdwrap/trt/*.h
    ${CMAKE_SOURCE_DIR}/src/3rdwrap/cuda/*.c*
    ${CMAKE_SOURCE_DIR}/src/3rdwrap/cuda/*.h
    ${CMAKE_SOURCE_DIR}/src/od/*.cpp
    ${CMAKE_SOURCE_DIR}/src/od/*.h
    ${CMAKE_SOURCE_DIR}/src/od/bodydetect_yolov5_ncnn/*.cpp
    ${CMAKE_SOURCE_DIR}/src/od/bodydetect_yolov5_ncnn/*.h
    ${CMAKE_SOURCE_DIR}/src/od/bodydetect_yolov5_trt/*.c*
    ${CMAKE_SOURCE_DIR}/src/od/bodydetect_yolov5_trt/*.h*

#    ${CMAKE_SOURCE_DIR}/src/od/facedetect_yolov5_trt/*.c*
#    ${CMAKE_SOURCE_DIR}/src/od/facedetect_yolov5_trt/*.h*

#    ${CMAKE_SOURCE_DIR}/src/od/headdetect_yolov5_trt/*.c*
#    ${CMAKE_SOURCE_DIR}/src/od/headdetect_yolov5_trt/*.h*

    ${CMAKE_SOURCE_DIR}/src/lm/*.cpp
    ${CMAKE_SOURCE_DIR}/src/lm/*.h
    ${CMAKE_SOURCE_DIR}/src/lm/pose_openpose_trt/*.c*
    ${CMAKE_SOURCE_DIR}/src/lm/pose_openpose_trt/*.h
#    ${CMAKE_SOURCE_DIR}/src/lm/*/openpose/*.c*
#    ${CMAKE_SOURCE_DIR}/src/lm/*/openpose/*.h
    ${CMAKE_SOURCE_DIR}/src/od/bodydetect_yolov5_3o_trt/*.c*
    ${CMAKE_SOURCE_DIR}/src/od/bodydetect_yolov5_3o_trt/*.h*
        )
    set(native_srcsakdbody 
                    ${native_srcs__tmp}
        )


    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")
    # include_directories(${OpenCV_INCLUDE_DIRS})
    my_add_library( AkdBody SHARED ${native_srcsakdbody} )
    set_property(TARGET AkdBody PROPERTY FOLDER "AkdBody")
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
    ADD_DEFINITIONS(-D_AMD64_)
    ADD_DEFINITIONS(-DOMP_WAIT_POLICY="passive")


    target_link_libraries( # Specifies the target library.
            AkdBody
            ncnn
            ${cvlib}
            #${CUBLAS_LIBRARIES}
            ${TRT_LIB}
            # ${OpenCV_LIBS}
            #${trtcvtm_LIB}
    )

   # target_include_directories(YzBody PRIVATE ${OpenCV_INCLUDE_DIRS})
    install(TARGETS AkdBody EXPORT AkdBody LIBRARY  DESTINATION lib)
 
        #install(FILES
            #${CMAKE_INSTALL_PREFIX}/lib/AkdBody.lib
            #${CMAKE_INSTALL_PREFIX}/bin/AkdBody.dll
            #DESTINATION yzbody/${VC}/lib
    #)
 

    #install(FILES
            #src/interface/body_d_c/BodyDetectClassy.h
            #src/YzComm.h
            #DESTINATION yzbody/include
        #)
        #install(FILES
            #demo/yzbodyi_test.cpp
            #DESTINATION yzbody
        #)



endif()

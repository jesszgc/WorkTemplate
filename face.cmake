if(AkdFACE)
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
    ${CMAKE_SOURCE_DIR}/src/fr/*.cpp
    ${CMAKE_SOURCE_DIR}/src/fr/*.h

    ${CMAKE_SOURCE_DIR}/src/fr/*/*.c*
    ${CMAKE_SOURCE_DIR}/src/fr/*/*.h*

    ${CMAKE_SOURCE_DIR}/src/od/*.cpp
    ${CMAKE_SOURCE_DIR}/src/od/*.h

    ${CMAKE_SOURCE_DIR}/src/od/facedetect_retinaface_ncnn/*.c*
    ${CMAKE_SOURCE_DIR}/src/od/facedetect_retinaface_ncnn/*.h*

    ${CMAKE_SOURCE_DIR}/src/od/facedetect_retinaface_trt/*.c*
    ${CMAKE_SOURCE_DIR}/src/od/facedetect_retinaface_trt/*.h*

    ${CMAKE_SOURCE_DIR}/src/interface/akdfacei/*.c*
    ${CMAKE_SOURCE_DIR}/src/interface/akdfacei/*.h*

        )
    set(native_srcsakdface 
                    ${native_srcs__tmp}
        )


    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "cmake")
    # include_directories(${OpenCV_INCLUDE_DIRS})
    my_add_library( AkdFace SHARED ${native_srcsakdface} )
    set_property(TARGET AkdFace PROPERTY FOLDER "AkdFace")
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
    ADD_DEFINITIONS(-D_AMD64_)
    ADD_DEFINITIONS(-DOMP_WAIT_POLICY="passive")


    target_link_libraries( # Specifies the target library.
            AkdFace
            ncnn
            ${cvlib}
            #${CUBLAS_LIBRARIES}
            ${TRT_LIB}
            # ${OpenCV_LIBS}
    )

   # target_include_directories(YzBody PRIVATE ${OpenCV_INCLUDE_DIRS})
    install(TARGETS AkdFace EXPORT AkdFace LIBRARY  DESTINATION lib)
 
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

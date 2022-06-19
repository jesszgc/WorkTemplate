
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

#if(OpenCV_FOUND)

# set(OpenCV_INCLUDE_DIRS "")
# set(OpenCV_LIBS "")





include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

include(../cmake/demo_utils.cmake)
 #    ncnn_add_example(frtest)
	# ncnn_add_example(fqtest)
 #    ncnn_add_example(hatclassifytest)
 #    ncnn_add_example(fronly)
    yzfacei_add_example(facedetect_corp_test)
    yzfacei_add_example(face_extractf)
    yzfacei_add_example(face_compare)
    yzfacei_add_example(fd_test)
 #    ncnn_add_example(yzfacetest)
 #    ncnn_add_example(posetest)
 #    ncnn_add_example(camTest)
 #    ncnn_add_example(odtest)
	# ncnn_add_example(fa_test)
    yzheadi_add_example(counttest)
    yzheadi_add_example(count_test)
    yzheadi_add_example(jiudian_test)
    yzcounti_add_example(yzcount_test)
 #    ncnn_add_example(ClassRoomFaceCheck_testcase)
 #    ncnn_add_example(DllTensorRtZoo_testMemcpy)
	# ncnn_add_example(iqa_test)
	# ncnn_add_example(fld_test)
 #    ncnn_add_example(BodyDetectClassy_test)
 #    ncnn_add_example(Classy_test)
 #    ncnn_add_example(BodyDetect_test)
	# feisi_add_example(feisi_test)
    yzfacei_add_example(yzfacei_test)
    yzfacei_add_example(yzfacei_test2)
    yzbodyi_add_example(yzbodyi_test)
    yzfacei_bodyi_add_example(yzfacei_bodyi_test)
    yzhelmeti_add_example(yzsafehati_test)
  #  yzocri_add_example(yzocri_test)
  if(${MSVC_TOOLSET_VERSION} MATCHES "142")
  yzocr_p_add_example(ocrpaddle_test)
  endif()
 # ncnn_add_example(DaSiamRPNtest)
    # ncnn_add_example(result_pick)
    # ncnn_add(face_judgetest_cv)
   ncnn_add(AllDataProcess_FaceImgClean)
#else()
 #   message(WARNING "OpenCV not found, examples won't be built")
#endif()

# add_subdirectory(camtest_cuda)







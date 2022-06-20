
#find_package(OpenCV REQUIRED)
#include_directories(${OpenCV_INCLUDE_DIRS})

#if(OpenCV_FOUND)

# set(OpenCV_INCLUDE_DIRS "")
# set(OpenCV_LIBS "")





include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../src)

include(../cmake/demo_utils.cmake)
     bodyi_add_example(bodytest)
     bodyi_add_example(posetest)

     facei_add_example(facetest)
     cv_add_example(cvtest)






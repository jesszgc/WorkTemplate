

                ## 判断编译器
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
                # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                # using GCC

                message(${CMAKE_CXX_COMPILER_ID})
                
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../3rd/opencv/lib/cmake/opencv4")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
                # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
                # using Visual Studio C++
 #set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../3rd/opencv/cv450_world_cuda114")

  #set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../3rd/opencv/cv450_world")  

endif()


find_package(OpenCV REQUIRED)
set(cvlib ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

macro(akdbodyi_add_example name)
    
    add_executable(${name} ${name}.cpp)
    target_compile_options(${name} PRIVATE -std=c++11)

    # include_directories( ${OpenCV_INCLUDE_DIRS})

    target_link_libraries(${name} PRIVATE AkdBody  ${cvlib})
    message(${cvlib})
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})

    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}../../)
    set_property(TARGET ${name} PROPERTY FOLDER "AkdBody")
    # add test to a virtual project group
    install(TARGETS ${name} RUNTIME DESTINATION bin)
endmacro()

macro(akdfacei_add_example name)

    add_executable(${name} ${name}.cpp)
    target_compile_options(${name} PRIVATE -std=c++11)

    # include_directories( ${OpenCV_INCLUDE_DIRS})

    target_link_libraries(${name} PRIVATE AkdFace  ${cvlib})
    message(${cvlib})
    target_include_directories(${name} PRIVATE ${OpenCV_INCLUDE_DIRS})

    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}../../)
    set_property(TARGET ${name} PROPERTY FOLDER "AkdFace")
    # add test to a virtual project group
    install(TARGETS ${name} RUNTIME DESTINATION bin)
endmacro()




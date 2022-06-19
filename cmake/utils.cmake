function(assign_source_group)
    foreach(_source IN ITEMS ${ARGN})
        if (IS_ABSOLUTE "${_source}")
            file(RELATIVE_PATH _source_rel "${CMAKE_CURRENT_SOURCE_DIR}" "${_source}")
        else()
            set(_source_rel "${_source}")
        endif()
        get_filename_component(_source_path "${_source_rel}" PATH)
        string(REPLACE "/" "\\" _source_path_msvc "${_source_path}")
        source_group("${_source_path_msvc}" FILES "${_source}")
    endforeach()
endfunction(assign_source_group)
 
function(my_add_library)
    foreach(_source IN ITEMS ${ARGN})
        assign_source_group(${_source})
    endforeach()
    # cuda_add
    cuda_add_library(${ARGV})

endfunction(my_add_library)


function(interface_add_library)
    foreach(_source IN ITEMS ${ARGN})
        assign_source_group(${_source})
    endforeach()
    # cuda_add

    cuda_add_library(${ARGV})
endfunction(interface_add_library)


if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        # using Clang
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        # using GCC

    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/../../3rd/AkdCommon_gcc")

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        # using Intel C++
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        # using Visual Studio C++

    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/../../3rd/YAkdCommon_MSVC")

    if(${MSVC_TOOLSET_VERSION} MATCHES "142")
      set(VC vc16)
    endif()
    if(${MSVC_TOOLSET_VERSION} MATCHES "140")
      set(VC vc14)
    endif()
endif()
   
















macro(add_tools name folderp)

    add_executable(${name} ${name}.cpp)
    target_compile_options(${name} PRIVATE -std=c++11)
    target_link_libraries(${name} PRIVATE ${OpenCV_LIBS})
    
    SET(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}../../)
    set_property(TARGET ${name} PROPERTY FOLDER ${folderp})

    install(TARGETS ${name} RUNTIME DESTINATION bin)
endmacro()

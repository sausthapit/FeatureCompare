cmake_minimum_required(VERSION 3.13)
project(FeatureCompare)

set(CMAKE_CXX_STANDARD 17)

#opencv
#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.1/)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lstdc++fs")
#set(OpenCV_DIR /home/saurav/local/installation/OpenCV-master/lib/cmake/opencv4)
#set(CMAKE_PREFIX_PATH /home/saurav/local/)
#set(OpenCV_DIR /usr/local/opencv240)
#set(CMAKE_PREFIX_PATH /usr/local/opencv240)
set(OpenCV_DIR /usr/local/opencv340)
set(CMAKE_PREFIX_PATH /usr/local/opencv340)
find_package( OpenCV 3.4 REQUIRED )

#vlfeat
#set(VLFEAT_DIR C:/dev/vlfeat-0.9.21/)
#set(VLFEAT_DIR /home/saurav/dev/vlfeat-0.9.21-bin/vlfeat-0.9.21/)
#set(VLFEAT_LIB_PATH ${VLFEAT_DIR}bin/glnxa64/)
#include_directories(${VLFEAT_DIR})
#message(${VLFEAT_LIB_PATH})
#find_library(VLFEAT_LIB NAMES vl HINTS ${VLFEAT_LIB_PATH})
#
#if (EXISTS ${VLFEAT_LIB})
#    message("-- VLFEAT libs: ${VLFEAT_LIB}")
#endif(EXISTS ${VLFEAT_LIB})


add_executable(FeatureCompare main.cpp OpenCVFeatures.cpp )

target_link_libraries(FeatureCompare ${OpenCV_LIBS} ${VLFEAT_LIB})

add_executable(ConvertHomographyFile ConvertHomography.cpp)
target_link_libraries(ConvertHomographyFile ${OpenCV_LIBS})
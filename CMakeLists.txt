cmake_minimum_required(VERSION 3.13)
project(FeatureCompare)

set(CMAKE_CXX_STANDARD 17)

#opencv
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-9.1/)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
#set(OpenCV_DIR /home/saurav/local/installation/OpenCV-master/lib/cmake/opencv4)
set(CMAKE_PREFIX_PATH /home/saurav/local/)
find_package( OpenCV REQUIRED )

#vlfeat
set(VLFEAT_DIR /media/saurav/Data/repos/vlfeat/)
set(VLFEAT_LIB_PATH ${VLFEAT_DIR}bin/glnxa64/)
include_directories(${VLFEAT_DIR})
find_library(VLFEAT_LIB NAMES vl HINTS ${VLFEAT_LIB_PATH})

if (EXISTS ${VLFEAT_LIB})
    message("-- VLFEAT libs: ${VLFEAT_LIB}")
endif(EXISTS ${VLFEAT_LIB})


add_executable(FeatureCompare main.cpp OpenCVFeatures.cpp OpenCVFeatures.h VLFeatures.cpp VLFeatures.h)

target_link_libraries(FeatureCompare ${OpenCV_LIBS} ${VLFEAT_LIB})
//This program detects various features from images and computes several descriptors

// Author: Saurav Sthapit

#include <iostream>
#include <experimental/filesystem>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "OpenCVFeatures.h"
//namespaces
namespace fs=std::experimental::filesystem;




int main() {
    OpenCVFeatures openCvFeatures;
    cv::Mat src=cv::imread("/media/saurav/Data/Datasets/KAIST/set00/images/set00/V000/visible/I00002.jpg");
    cv::Mat mask=cv::Mat();
    OCVKeypoints ocvKeypoints;
    OCVDescriptors ocvDescriptors;
    openCvFeatures.runDetectors(src,mask,ocvKeypoints);
    openCvFeatures.extractDescriptors(src,ocvKeypoints,ocvDescriptors);
    return 0;
}
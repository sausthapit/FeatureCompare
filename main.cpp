//This program detects various features from images and computes several descriptors

// Author: Saurav Sthapit

#include <iostream>
#include <experimental/filesystem>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include "OpenCVFeatures.h"
#include "QualityEvaluator.h"
//namespaces
namespace fs=std::experimental::filesystem;
using namespace cv;



int main() {
    OpenCVFeatures openCvFeatures;
//    cv::Mat src=cv::imread("/media/saurav/Data/Datasets/KAIST/set00/images/set00/V000/visible/I00002.jpg");
//    cv::Mat src=cv::imread("/home/saurav/dev/datasets/TUM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png");
//    cv::Mat mask=cv::Mat();
//    cv::Mat out;
//    OCVKeypoints ocvKeypoints;
//    OCVDescriptors ocvDescriptors;
////    cv::imshow("test",src);
////    cv::waitKey(0);
//    cvtColor( src, src, cv::COLOR_BGR2GRAY );
    cv::Ptr<FeatureDetector> sift=cvx2d::SIFT::create(); // SIFT
    Ptr<BaseQualityEvaluator> evals[] =
            {

                    new DetectorQualityEvaluator(sift,"FAST", "quality-detector-fast")
            };
    for( size_t i = 0; i < sizeof(evals)/sizeof(evals[0]); i++ )
    {
        evals[i]->run();
        cout << endl;
    }

//    openCvFeatures.runDetectors(src,mask,ocvKeypoints);
//    openCvFeatures.cornerHarris(src,out,ocvKeypoints);
//    openCvFeatures.extractDescriptors(src,ocvKeypoints,ocvDescriptors);
    return 0;
}
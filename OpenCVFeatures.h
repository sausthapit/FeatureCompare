//
// Created by saurav on 19/07/19.
//

#ifndef FEATURECOMPARE_OPENCVFEATURES_H
#define FEATURECOMPARE_OPENCVFEATURES_H

#include <map>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/imgproc.hpp>

namespace cvx2d= cv::xfeatures2d;
typedef std::map<std::string,cv::Ptr<cv::FeatureDetector>> OCVFeatureDetectors;
typedef std::map<std::string,cv::Ptr<cv::DescriptorExtractor>> OCVDescriptorExtractor;


typedef std::map<std::string,std::vector<cv::KeyPoint>> OCVKeypoints;
typedef std::map<std::pair<std::string,std::string>,cv::Mat> OCVDescriptors;

class OpenCVFeatures {
private:
    OCVFeatureDetectors detectorList;
    OCVDescriptorExtractor descriptorList;
    OCVFeatureDetectors newDetectorList;

    static void unpackOctave(const cv::KeyPoint& , int& , int& , float& );
    cv::FeatureDetector featureDetector;

//    Harris parameters
    int blockSize_H=2;
    int apertureSize_H=3;
    double k_H =0.04;
    int thresh_H=200;
public:
    OpenCVFeatures();
    void runDetectors(cv::Mat, cv::Mat, OCVKeypoints&);
    void extractDescriptors(cv::Mat,OCVKeypoints&,OCVDescriptors&);
    void cornerHarris(cv::Mat,cv::Mat, OCVKeypoints&);



};


#endif //FEATURECOMPARE_OPENCVFEATURES_H

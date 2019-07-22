//
// Created by saurav on 19/07/19.
//

#ifndef FEATURECOMPARE_OPENCVFEATURES_H
#define FEATURECOMPARE_OPENCVFEATURES_H

#include <map>
#include <opencv2/xfeatures2d/nonfree.hpp>
namespace cvx2d= cv::xfeatures2d;
typedef std::map<std::string,cv::Ptr<cv::Feature2D>> OCVFeatures;
typedef std::map<std::string,std::vector<cv::KeyPoint>> OCVKeypoints;
typedef std::map<std::pair<std::string,std::string>,cv::Mat> OCVDescriptors;

class OpenCVFeatures {
private:
    OCVFeatures detectorList,descriptorList;
    static void unpackOctave(const cv::KeyPoint& , int& , int& , float& );
public:
    OpenCVFeatures();
    void runDetectors(cv::Mat, cv::Mat, OCVKeypoints&);
    void extractDescriptors(cv::Mat,OCVKeypoints&,OCVDescriptors&);




};


#endif //FEATURECOMPARE_OPENCVFEATURES_H

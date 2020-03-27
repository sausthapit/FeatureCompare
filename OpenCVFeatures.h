//
// Created by saurav on 19/07/19.
//

#ifndef FEATURECOMPARE_OPENCVFEATURES_H
#define FEATURECOMPARE_OPENCVFEATURES_H

#include <map>
#include<opencv2/features2d.hpp>
#if CV_MAJOR_VERSION == 2

#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/legacy/legacy.hpp"

#else
#define NONFREE
#ifdef NONFREE
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
namespace cvx2d = cv::xfeatures2d;
using namespace cv;
#endif // NONFREE
#endif //cv 2
#include "opencv2/calib3d.hpp"

#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
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
//    Ptr<cv::FeatureDetector> featureDetector;

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
	void evaluateDetectors(const cv::Mat&, const  cv::Mat&, const cv::Mat&, OCVKeypoints&, OCVKeypoints&,std::map<std::string,float>&, std::map<std::string,int>& );
    cv::Mat estimateHomography(cv::Mat img_object,cv::Mat img_scene,cv::Mat& img_matches);
    void evaluateDescriptorMatcher( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                                    std::vector<KeyPoint>& keypoints1, std::vector<KeyPoint>& keypoints2,
                                                    std::vector<std::vector<DMatch> >* _matches1to2, std::vector<std::vector<uchar> >* _correctMatches1to2Mask,
                                                    std::vector<Point2f>& recallPrecisionCurve,
                                                    const Ptr<DescriptorMatcher>& _dmatcher );
};


#endif //FEATURECOMPARE_OPENCVFEATURES_H

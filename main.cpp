//This program detects various features from images and computes several descriptors

// Author: Saurav Sthapit
#define DEBUG
#include <iostream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <experimental/filesystem>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/xfeatures2d.hpp"

#include "OpenCVFeatures.h"
#include "QualityEvaluator.h"
#include "VLFeatures.h"


//namespaces
namespace fs=std::experimental::filesystem;

using namespace cv;



int main() {
	std::string dirname = "G:/dataset/oxford_affine/bikes/";
    OpenCVFeatures openCvFeatures;
    cv::Mat src1=cv::imread("G:/dataset/oxford_affine/bikes/img1.ppm");
	cv::Mat src2 = cv::imread("G:/dataset/oxford_affine/bikes/img2.ppm");
//    cv::Mat src=cv::imread("/home/saurav/dev/datasets/TUM/rgbd_dataset_freiburg1_xyz/rgb/1305031102.175304.png");
    cv::Mat mask=cv::Mat();
    cv::Mat out;
    OCVKeypoints ocvKeypoints,ocvKeypoints2;
    OCVDescriptors ocvDescriptors, ocvDescriptors2;
	VLFeatures vl;
////    cv::imshow("test",src);
////    cv::waitKey(0);
//    cvtColor( src, src, cv::COLOR_BGR2GRAY );
//    cv::Ptr<FeatureDetector> sift=cvx2d::SIFT::create(); // SIFT
//    cv::Ptr<FeatureDetector> fast=cv::FastFeatureDetector::create(); // SIFT
//    cv::Ptr<FeatureDetector> sift=cvx2d::SiftFeatureDetector.create(); // SIFT

    //cv::Ptr<FeatureDetector> sift=cvx2d::SIFT::create(); // SIFT
    //cv::Ptr<FeatureDetector> surf=cvx2d::SURF::create(); // SURF

#ifdef OLD


    cv::Ptr<FeatureDetector> gfft=cv::GFTTDetector::create(); // Good Features to Track
    cv::Ptr<FeatureDetector> fast=cv::FastFeatureDetector::create(); // Fast
    cv::Ptr<FeatureDetector> brisk=cv::BRISK::create(); // BRISK
    cv::Ptr<FeatureDetector> akaze=cv::AKAZE::create(); // AKAZE

    Ptr<DescriptorExtractor> harris_2d=GFTTDetector::create();
    Ptr<DescriptorExtractor> simple=SimpleBlobDetector::create();
    Ptr<DescriptorExtractor> mser =MSER::create();
    //Ptr<DescriptorExtractor> star=cvx2d::StarDetector::create();

    Ptr<BaseQualityEvaluator> evals[] =
            {

                    new DetectorQualityEvaluator(fast,"FAST", "quality-detector-fast"),
                    //new DetectorQualityEvaluator(sift,"sift", "quality-detector-sift"),
                    //new DetectorQualityEvaluator(surf,"surf", "quality-detector-surf"),
                    new DetectorQualityEvaluator(gfft,"gfft", "quality-detector-gfft"),
                    new DetectorQualityEvaluator(brisk,"brisk", "quality-detector-brisk"),
                    new DetectorQualityEvaluator(akaze,"akaze", "quality-detector-akaze"),

                    //new De




            };
    for( size_t i = 0; i < sizeof(evals)/sizeof(evals[0]); i++ )
    {
        evals[i]->run();
        cout << endl;
    }
#endif 
#define NEW
#ifdef NEW
	openCvFeatures.runDetectors(src1, mask, ocvKeypoints);
	//vl.extractLIOP(src1, ocvKeypoints,ocvDescriptors);
	//openCvFeatures.extractDescriptors(src1,ocvKeypoints,ocvDescriptors);

	openCvFeatures.runDetectors(src2, mask, ocvKeypoints2);
	//vl.extractLIOP(src2, ocvKeypoints2, ocvDescriptors2);
	//openCvFeatures.extractDescriptors(src2, ocvKeypoints2, ocvDescriptors2);
	stringstream filename; filename << "H1to" << "2" << "p.xml";

	FileStorage fs(dirname + filename.str(), FileStorage::READ);
	if (!fs.isOpened())
	{
		cout << "filename " << dirname + filename.str() << endl;
		FileStorage fs2(dirname + filename.str(), FileStorage::READ);
		return false;
	}
	cv::Mat H1to2;
	fs.getFirstTopLevelNode() >> H1to2;
	std::map<std::string, float> repeatability;
	std::map<std::string, int> correspCount;
	openCvFeatures.evaluateDetectors(src1, src2, H1to2, ocvKeypoints, ocvKeypoints2, repeatability, correspCount);

#endif // NEW

   

    return 0;
}
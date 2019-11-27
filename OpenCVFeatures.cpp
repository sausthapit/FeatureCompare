//
// Created by saurav on 19/07/19.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include "OpenCVFeatures.h"

using namespace cv;
OpenCVFeatures::OpenCVFeatures() {


	Ptr<FeatureDetector> gfft = GFTTDetector::create(); // Good Features to Track
	Ptr<FeatureDetector> fast = FastFeatureDetector::create(); // Fast
	Ptr<FeatureDetector> brisk = BRISK::create(); // BRISK
	Ptr<FeatureDetector> akaze = AKAZE::create(); // AKAZE

	Ptr<FeatureDetector> orb = ORB::create();
	// Todo: harris detector option enable
	Ptr<FeatureDetector> harris_2d = GFTTDetector::create();
	Ptr<FeatureDetector> simple = SimpleBlobDetector::create();
	Ptr<FeatureDetector> mser = MSER::create();


	// Detectors

#ifdef NONFREE
	Ptr<FeatureDetector> sift = cvx2d::SIFT::create(); // SIFT
	Ptr<FeatureDetector> surf = cvx2d::SURF::create(); // SURF
	Ptr<FeatureDetector> star = cvx2d::StarDetector::create();

	detectorList["sift"] = (sift);
	detectorList["surf"] = (surf);
	detectorList["star"] = (star);
	descriptorList["sift"] = sift;
	descriptorList["surf"] = surf;
#endif // NONFREE


	detectorList["gfft"] = (gfft);
	detectorList["fast"] = (fast);
	detectorList["brisk"] = (brisk);
	detectorList["akaze"] = (akaze);
	detectorList["harris_2d"] = (harris_2d);
	detectorList["simple"] = (simple);
	detectorList["mser"] = (mser);





	descriptorList["orb"] = orb;
	//descriptorList["akaze"]=akaze;
	descriptorList["brisk"] = brisk;
}
void OpenCVFeatures::runDetectors(Mat image, Mat mask, OCVKeypoints& keypointsWithNames) {


	for_each(this->detectorList.begin(), this->detectorList.end(), [image, mask, &keypointsWithNames](const std::pair<std::string, cv::Ptr<cv::FeatureDetector>>& fd) {

		std::cout << "========" << fd.first << " ";
		std::vector<KeyPoint> kp;
		Mat imgKeypoints;
		fd.second->detect(image, kp, mask);


		std::vector<KeyPoint>::iterator i = kp.begin();

		while (i != kp.end()) {
			int layer;
			float scale;
			int octave;

			unpackOctave(i.operator*(), octave, layer, scale);
			if (octave < 0) {
				//                std::cout << "removing one keypoint" << std::endl;
				i = kp.erase(i);
			}
			else {
				i.operator*().octave = octave;
				++i;
			}
		}
		keypointsWithNames[fd.first] = kp;
		std::cout << "Keypoints detected" << kp.size() << "========" << std::endl;
		});
}



void OpenCVFeatures::unpackOctave(const KeyPoint& kpt, int& octave, int& layer, float& scale)
{
	octave = kpt.octave & 255;
	layer = (kpt.octave >> 8) & 255;
	octave = octave < 128 ? octave : (-128 | octave);
	scale = octave >= 0 ? 1.f / (1 << octave) : (float)(1 << -octave);
}
void OpenCVFeatures::extractDescriptors(Mat image, OCVKeypoints& ocvKeypoints, OCVDescriptors& ocvDescriptors) {
	for_each(ocvKeypoints.begin(), ocvKeypoints.end(), [image, &ocvDescriptors, this](const std::pair<std::string, std::vector<cv::KeyPoint>>& ocvKeypoint) {
		std::string det = ocvKeypoint.first;
		std::vector<KeyPoint> kp = ocvKeypoint.second;
		for_each(this->descriptorList.begin(), this->descriptorList.end(), [image, kp, det, &ocvDescriptors](const std::pair<std::string, cv::Ptr<cv::DescriptorExtractor>>& fdesc) {
			//            std::cout<<fdesc.first<<std::endl;
			std::pair<std::string, std::string> det_desc = std::pair(det, fdesc.first);
			std::cout << "Detector:" << det_desc.first << " Descriptor: " << det_desc.second << std::endl;
			Mat desc;
			std::vector<KeyPoint> keypoints = kp;
			Ptr<Feature2D > tmp = fdesc.second;
			tmp->compute(image, keypoints, desc);



			ocvDescriptors[det_desc] = desc;
			});
		});
}

void OpenCVFeatures::cornerHarris(Mat image, Mat mask, OCVKeypoints& ocvKeypoints) {
	Mat dst = Mat::zeros(image.size(), CV_32FC1);
	cv::cornerHarris(image, dst, blockSize_H, apertureSize_H, k_H);
	Mat dst_norm, dst_norm_scaled;
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, mask);
	convertScaleAbs(dst_norm, dst_norm_scaled);
	std::vector<KeyPoint> keypoints;
	for (int i = 0; i < dst_norm.rows; i++)
	{
		for (int j = 0; j < dst_norm.cols; j++)
		{
			if ((int)dst_norm.at<float>(i, j) > thresh_H)
			{
				KeyPoint kp = KeyPoint();
				kp.pt.x = i;
				kp.pt.y = j;
				kp.octave = 0;
				kp.size = 3;
				keypoints.emplace_back(kp);
			}
		}
	}
	ocvKeypoints["harris"] = keypoints;
}

void OpenCVFeatures::evaluateDetectors(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& H1to2, OCVKeypoints& kp1, OCVKeypoints& kp2, std::map<std::string, float>& repeatability, std::map<std::string, int>& correspCount)
{	
	std::for_each(kp1.begin(), kp1.end(), [img1, img2, H1to2, kp2,&repeatability, &correspCount](std::pair<std::string, std::vector<cv::KeyPoint>> keyPoint1)mutable {
		std::for_each(kp2.begin(), kp2.end(), [img1, img2, H1to2, keyPoint1, &repeatability, &correspCount](std::pair<std::string, std::vector<cv::KeyPoint>> keyPoint2) mutable {
			if (keyPoint1.first == keyPoint2.first)
			{
				std::vector<cv::KeyPoint> tmp1 = keyPoint1.second;
				std::vector<cv::KeyPoint> tmp2 = keyPoint2.second;
				float rep;
				int count;
				cv::Ptr<cv::FeatureDetector> fd;
				evaluateFeatureDetector(img1, img2, H1to2, &tmp1, &tmp2, rep, count, fd);
				repeatability[keyPoint1.first] = rep;
				correspCount[keyPoint1.first] = count;
			}
			});

		});

	std::cout << repeatability.size();
}

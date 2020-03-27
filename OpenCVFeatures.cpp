//
// Created by saurav on 19/07/19.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include "OpenCVFeatures.h"

using namespace cv;
OpenCVFeatures::OpenCVFeatures() {


//
//	Ptr<FeatureDetector> gfft = GFTTDetector::create(); // Good Features to Track
//	Ptr<FeatureDetector> fast = FastFeatureDetector::create(); // Fast
//	Ptr<FeatureDetector> brisk = BRISK::create(); // BRISK
//	Ptr<FeatureDetector> akaze = AKAZE::create(); // AKAZE
//
//	Ptr<FeatureDetector> orb = ORB::create();
//	// Todo: harris detector option enable
//	Ptr<FeatureDetector> harris_2d = GFTTDetector::create();
//	Ptr<FeatureDetector> simple = SimpleBlobDetector::create();
//	Ptr<FeatureDetector> mser = MSER::create();
    Ptr<FeatureDetector> fast =FastFeatureDetector::create(); // Fast
//    Ptr<FeatureDetector> surf = FastFeatureDetector::create(); // Fast
    Ptr<DescriptorExtractor> orb = ORB::create();
	// Detectors

#ifdef NONFREE
	Ptr<FeatureDetector> sift = cvx2d::SIFT::create(); // SIFT
	Ptr<FeatureDetector> surf = cvx2d::SURF::create(); // SURF
	Ptr<FeatureDetector> star = cvx2d::StarDetector::create();

//	detectorList["sift"] = (sift);
//	detectorList["surf"] = (surf);
//	detectorList["star"] = (star);
//	descriptorList["sift"] = sift;
//	descriptorList["surf"] = surf;
#endif // NONFREE


//	detectorList["gfft"] = (gfft);
	detectorList["fast"] = (fast);
//	detectorList["brisk"] = (brisk);
//	detectorList["akaze"] = (akaze);
//	detectorList["harris_2d"] = (harris_2d);
//	detectorList["simple"] = (simple);
//	detectorList["mser"] = (mser);





	descriptorList["orb"] = orb;
	//descriptorList["akaze"]=akaze;
//	descriptorList["brisk"] = brisk;
}
void OpenCVFeatures::runDetectors(Mat image, Mat mask, OCVKeypoints& keypointsWithNames) {


	for_each(this->detectorList.begin(), this->detectorList.end(), [image, mask, &keypointsWithNames](const std::pair<std::string, cv::Ptr<cv::FeatureDetector>>& fd) {

		std::cout << "========" << fd.first << " ";
		std::vector<KeyPoint> kp;
		Mat imgKeypoints;
//		fd.second->detect(image, kp);
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

//	std::cout << repeatability.size();
}
Mat OpenCVFeatures::estimateHomography(Mat img_object,Mat img_scene, Mat& img_matches){
    if ( img_object.empty() || img_scene.empty() )
    {
        std::cout << "Could not open or find the image!\n" << std::endl;
//        parser.printMessage();
//        return -1;
    }
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    int minHessian = 400;
    Ptr<FeatureDetector> detector = cvx2d::SURF::create( minHessian );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;
    Mat descriptors_object, descriptors_scene;

//    detector->detect(img_object,keypoints_object);
//    detector->detect(img_scene,keypoints_scene);
//    desc->compute(img_object,keypoints_object,descriptors_object);
//    desc->compute(img_scene,keypoints_scene,descriptors_scene);
    detector->detectAndCompute( img_object, noArray(), keypoints_object, descriptors_object );
    detector->detectAndCompute( img_scene, noArray(), keypoints_scene, descriptors_scene );
    //-- Step 2: Matching descriptor vectors with a FLANN based matcher
    // Since SURF is a floating-point descriptor NORM_L2 is used
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
//    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("FlannBasedMatcher");
    std::vector< std::vector<DMatch> > knn_matches;
    matcher->knnMatch( descriptors_object, descriptors_scene, knn_matches, 2 );
    //-- Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
//    Mat img_matches;
    drawMatches( img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, Scalar::all(-1),
                 Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    //-- Localize the object
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;
    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( obj, scene, RANSAC );
//    -- Get the corners from the image_1 ( the object to be "detected" )
    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = Point2f(0, 0);
    obj_corners[1] = Point2f( (float)img_object.cols, 0 );
    obj_corners[2] = Point2f( (float)img_object.cols, (float)img_object.rows );
    obj_corners[3] = Point2f( 0, (float)img_object.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);
    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
//    line( img_matches, scene_corners[0] + Point2f((float)img_object.cols, 0),
//          scene_corners[1] + Point2f((float)img_object.cols, 0), Scalar(0, 255, 0), 4 );
//    line( img_matches, scene_corners[1] + Point2f((float)img_object.cols, 0),
//          scene_corners[2] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[2] + Point2f((float)img_object.cols, 0),
//          scene_corners[3] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
//    line( img_matches, scene_corners[3] + Point2f((float)img_object.cols, 0),
//          scene_corners[0] + Point2f((float)img_object.cols, 0), Scalar( 0, 255, 0), 4 );
    //-- Show detected matches
    imshow("Good Matches", img_matches );
    waitKey(1);
    return H;

}
/*
void OpenCVFeatures::evaluateDescriptorMatcher( const Mat& img1, const Mat& img2, const Mat& H1to2,
                                           std::vector<KeyPoint>& keypoints1, std::vector<KeyPoint>& keypoints2,
                                                       std::vector<std::vector<DMatch> >* _matches1to2, std::vector<std::vector<uchar> >* _correctMatches1to2Mask,
                                                       std::vector<Point2f>& recallPrecisionCurve,
                                           const Ptr<DescriptorMatcher>& _dmatcher )
{
    Ptr<DescriptorMatcher> dmatcher = _dmatcher;
    dmatcher->clear();

    std::vector<std::vector<DMatch> > *matches1to2, buf1;
    matches1to2 = _matches1to2 != 0 ? _matches1to2 : &buf1;

    std::vector<std::vector<uchar> > *correctMatches1to2Mask, buf2;
    correctMatches1to2Mask = _correctMatches1to2Mask != 0 ? _correctMatches1to2Mask : &buf2;

    if( keypoints1.empty() )
        CV_Error( CV_StsBadArg, "keypoints1 must not be empty" );

    if( matches1to2->empty() && dmatcher.empty() )
        CV_Error( CV_StsBadArg, "dmatch must not be empty when matches1to2 is empty" );

    bool computeKeypoints2ByPrj = keypoints2.empty();
    if( computeKeypoints2ByPrj )
    {
        assert(0);
        // TODO: add computing keypoints2 from keypoints1 using H1to2
    }

    if( matches1to2->empty() || computeKeypoints2ByPrj )
    {
        dmatcher->clear();
        dmatcher->radiusMatch( img1, keypoints1, img2, keypoints2, *matches1to2, std::numeric_limits<float>::max() );
    }
    float repeatability;
    int correspCount;
    Mat thresholdedOverlapMask; // thresholded allOverlapErrors
    calculateRepeatability( img1, img2, H1to2, keypoints1, keypoints2, repeatability, correspCount, &thresholdedOverlapMask );


    correctMatches1to2Mask->resize(matches1to2->size());
    for( size_t i = 0; i < matches1to2->size(); i++ )
    {
        (*correctMatches1to2Mask)[i].resize((*matches1to2)[i].size());
        for( size_t j = 0;j < (*matches1to2)[i].size(); j++ )
        {
            int indexQuery = (*matches1to2)[i][j].queryIdx;
            int indexTrain = (*matches1to2)[i][j].trainIdx;
            (*correctMatches1to2Mask)[i][j] = thresholdedOverlapMask.at<uchar>( indexQuery, indexTrain );
        }
    }

    computeRecallPrecisionCurve( *matches1to2, *correctMatches1to2Mask, recallPrecisionCurve );
}
 */

void OpenCVFeatures::evaluateDescriptorMatcher(const Mat &img1, const Mat &img2, const Mat &H1to2,
                                               std::vector<KeyPoint> &keypoints1, std::vector<KeyPoint> &keypoints2,
                                               std::vector<std::vector<DMatch> > *_matches1to2,
                                               std::vector<std::vector<uchar> > *_correctMatches1to2Mask,
                                               std::vector<Point2f> &recallPrecisionCurve,
                                               const Ptr<DescriptorMatcher> &_dmatcher) {
    Ptr<DescriptorMatcher> dmatcher = _dmatcher;
    dmatcher->clear();

    vector<vector<DMatch> > *matches1to2, buf1;
    matches1to2 = _matches1to2 != 0 ? _matches1to2 : &buf1;

    vector<vector<uchar> > *correctMatches1to2Mask, buf2;
    correctMatches1to2Mask = _correctMatches1to2Mask != 0 ? _correctMatches1to2Mask : &buf2;

    if( keypoints1.empty() )
        CV_Error( CV_StsBadArg, "keypoints1 must not be empty" );

    if( matches1to2->empty() && dmatcher.empty() )
        CV_Error( CV_StsBadArg, "dmatch must not be empty when matches1to2 is empty" );

    bool computeKeypoints2ByPrj = keypoints2.empty();
    if( computeKeypoints2ByPrj )
    {
        assert(0);
        // TODO: add computing keypoints2 from keypoints1 using H1to2
    }

    if( matches1to2->empty() || computeKeypoints2ByPrj )
    {
        dmatcher->clear();
        dmatcher->radiusMatch(  keypoints1, keypoints2, *matches1to2, std::numeric_limits<float>::max() );
    }
    float repeatability;
    int correspCount;
    Mat thresholdedOverlapMask; // thresholded allOverlapErrors
//    calculateRepeatability( img1, img2, H1to2, keypoints1, keypoints2, repeatability, correspCount, &thresholdedOverlapMask );

    correctMatches1to2Mask->resize(matches1to2->size());
    for( size_t i = 0; i < matches1to2->size(); i++ )
    {
        (*correctMatches1to2Mask)[i].resize((*matches1to2)[i].size());
        for( size_t j = 0;j < (*matches1to2)[i].size(); j++ )
        {
            int indexQuery = (*matches1to2)[i][j].queryIdx;
            int indexTrain = (*matches1to2)[i][j].trainIdx;
            (*correctMatches1to2Mask)[i][j] = thresholdedOverlapMask.at<uchar>( indexQuery, indexTrain );
        }
    }

    computeRecallPrecisionCurve( *matches1to2, *correctMatches1to2Mask, recallPrecisionCurve );

}





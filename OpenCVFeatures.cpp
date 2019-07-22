//
// Created by saurav on 19/07/19.
//

#include <iostream>
#include <opencv2/highgui.hpp>
#include "OpenCVFeatures.h"

OpenCVFeatures::OpenCVFeatures() {

    cv::Ptr<cvx2d::SIFT> sift=cvx2d::SIFT::create(); // SIFT
    cv::Ptr<cvx2d::SURF> surf=cvx2d::SURF::create(); // SURF
    cv::Ptr<cv::GFTTDetector> gfft=cv::GFTTDetector::create(); // Good Features to Track
    cv::Ptr<cv::FastFeatureDetector> fast=cv::FastFeatureDetector::create(); // Fast
    cv::Ptr<cv::BRISK> brisk=cv::BRISK::create(); // BRISK
    cv::Ptr<cv::AKAZE> akaze=cv::AKAZE::create(); // AKAZE

    cv::Ptr<cv::ORB> orb=cv::ORB::create();
// Detectors
    detectorList["sift"]=(sift);
    detectorList["surf"]=(surf);
    detectorList["gfft"]=(gfft);
    detectorList["fast"]=(fast);
    detectorList["brisk"]=(brisk);
    detectorList["akaze"]=(akaze);

// Descriptors
    descriptorList["sift"]=sift;
    descriptorList["surf"]=surf;
    descriptorList["orb"]=orb;
//        descriptorList["akaze"]=akaze;
    descriptorList["brisk"]=brisk;
}
void OpenCVFeatures::runDetectors(cv::Mat image,cv::Mat mask, OCVKeypoints& keypointsWithNames) {

    for_each( this->detectorList.begin(),this->detectorList.end(),[image,mask,&keypointsWithNames](const auto &fd){

        std::cout<<"========"<< fd.first<<" ";
        std::vector<cv::KeyPoint> kp;
        cv::Mat imgKeypoints;
        fd.second->detect(image,kp,mask);


        std::vector<cv::KeyPoint>::iterator i = kp.begin();

        while (i != kp.end()) {
            int layer;
            float scale;
            int octave;

            unpackOctave(i.operator*(),octave,layer,scale);
            if (octave<0) {
//                std::cout << "removing one keypoint" << std::endl;
                i = kp.erase(i);
            }
            else {
                i.operator*().octave=octave;
                ++i;
            }
        }
        keypointsWithNames[fd.first]=kp;
        std::cout<<"Keypoints detected"<<kp.size()<<"========"<<std::endl;
    });
}



void OpenCVFeatures::unpackOctave(const cv::KeyPoint& kpt, int& octave, int& layer, float& scale)
{
    octave = kpt.octave & 255;
    layer = (kpt.octave >> 8) & 255;
    octave = octave < 128 ? octave : (-128 | octave);
    scale = octave >= 0 ? 1.f/(1 << octave) : (float)(1 << -octave);
}
void OpenCVFeatures::extractDescriptors(cv::Mat image, OCVKeypoints &ocvKeypoints,OCVDescriptors &ocvDescriptors) {
    for_each(ocvKeypoints.begin(),ocvKeypoints.end(),[image,&ocvDescriptors,this](const auto &ocvKeypoint){
        std::string det=ocvKeypoint.first;
        std::vector<cv::KeyPoint> kp=ocvKeypoint.second;
        for_each(this->descriptorList.begin(),this->descriptorList.end(),[image,kp,det,&ocvDescriptors](const auto &fdesc){
//            std::cout<<fdesc.first<<std::endl;
            std::pair<std::string,std::string> det_desc=std::pair(det,fdesc.first);
            std::cout<<"Detector:"<<det_desc.first<<" Descriptor: "<<det_desc.second<<std::endl;
            cv::Mat desc;
            std::vector<cv::KeyPoint> keypoints=kp;
            cv::Ptr<cv::Feature2D > tmp=fdesc.second;
            tmp->compute(image,keypoints,desc);

            ocvDescriptors[det_desc]=desc;
        });
    });
}
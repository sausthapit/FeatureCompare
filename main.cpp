//This program detects various features from images and computes several descriptors

// Author: Saurav Sthapit
#define DEBUG
#include <iostream>
#include <fstream>
#include<iomanip>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING

#include <experimental/filesystem>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/xfeatures2d.hpp"

#include "OpenCVFeatures.h"
#include "QualityEvaluator.h"
//#include "VLFeatures.h"


//namespaces
namespace fs=std::experimental::filesystem;


using namespace cv;
//global options

string rootDir;
int cam;
int equalise_option;
int convert16to8bit;

vector<Mat> images;
vector<OCVKeypoints> ocvKeypoints;
vector<OCVDescriptors> ocvDescriptors;
vector<std::map<std::string, float>> repeatabilities;
vector<std::map<std::string, int>> correspCounts;
void LoadImages(const string &dir, const int cam, vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimestamps){
    ifstream f;
    f.open(rootDir+"/image_data.csv");
    string line;
    bool first= true;
    while(!f.eof()) {
        getline(f, line);
        double startTime=0;

        if(!line.empty())
        {
            stringstream ss;
            ss << line;
            int count;
            ss>> count;
            double timestamp;
            string tmp;
            getline(ss,tmp,',');
            getline(ss,tmp,',');

//            ss>>tmp;
            timestamp=std::stod(tmp.c_str())/1e9;
//            std::cout << tmp << std::endl;

            if(first)
            {
                startTime=timestamp;
                first=false;
            }
            vTimestamps.push_back(timestamp-startTime);
            stringstream fnameleftss,fnamerightss;
            fnameleftss<<dir<<"/cam"<<cam<<"_image"<<std::setw(5) << std::setfill('0')<<count<<".png";
            fnamerightss<<dir<<"/cam"<<cam+1<<"_image"<<std::setw(5) << std::setfill('0')<<count<<".png";
            vstrImageLeft.push_back("/"+fnameleftss.str());
            vstrImageRight.push_back("/"+fnamerightss.str());
        }
    }
}



void preprocess(Mat& img);

void preprocess(Mat &img) {
    if (img.channels() > 2)
        cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);

    if (convert16to8bit == 1) {
        double m = 30.0 / 64.0;
        double mean_imgLeft = cv::mean(img)[0];
        img = img * m + (127 - mean_imgLeft * m);
        img.convertTo(img, CV_8U);

    }

    if (equalise_option == 1) {
        cv::equalizeHist(img, img);

    } else if (equalise_option == 2) {
        cv::Ptr<cv::CLAHE> ptr = cv::createCLAHE();
        ptr->setClipLimit(2);
        ptr->setTilesGridSize(cv::Size(8, 8));
        ptr->apply(img, img);
    }
}

int main(int argc, char **argv) {
//    std::string dirname = "G:/dataset/oxford_affine/bikes/";
    rootDir=string(argv[1]);
    cam=atoi(argv[2]);
    equalise_option=atoi(argv[3]);
    convert16to8bit=atoi(argv[4]);
    OpenCVFeatures openCvFeatures;
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages("",cam, vstrImageLeft, vstrImageRight, vTimestamps);
//    const int nImages = vstrImageLeft.size();
    const int nImages = 10;
    cv::Mat imgNext;
    cv::Mat img;
    cv::Mat mask=cv::Mat();
    cv::Mat out;

//    VLFeatures vl;


    for(int ni=0; ni<nImages; ni++) {
        // Read left and right images from file
        cout << "Image:" << ni << endl;
        img = cv::imread(rootDir + vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED|CV_LOAD_IMAGE_ANYDEPTH);

        if (img.empty()) {
            cerr << endl << "Failed to load image at: " << rootDir
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        preprocess(img);

        OCVKeypoints temp_ocvKeypoints;
        OCVDescriptors temp_ocvDescriptors;
        openCvFeatures.runDetectors(img, mask, temp_ocvKeypoints);
//        vl.extractLIOP(img, temp_ocvKeypoints,temp_ocvDescriptors);
        openCvFeatures.extractDescriptors(img,temp_ocvKeypoints,temp_ocvDescriptors);

        images.emplace_back(img);
        ocvKeypoints.emplace_back(temp_ocvKeypoints);
        ocvDescriptors.emplace_back(temp_ocvDescriptors);


    }
    VideoWriter video("image_matches"+to_string(cam)+".avi",CV_FOURCC('M','J','P','G'),10, Size(img.cols*2,img.rows),true);

    for(int ni=0; ni<nImages-1; ni++) {
        Mat img_matches;
//        estimate homography based on SURF features and RANSAC.
        cv::Mat H = openCvFeatures.estimateHomography(images[ni], images[ni+1],img_matches);
        std::map<std::string, float> repeatability;
        std::map<std::string, int> correspCount;

//        cout<<ni<<endl;
        video.write(img_matches);
        openCvFeatures.evaluateDetectors(images[ni], images[ni+1], H, ocvKeypoints[ni], ocvKeypoints[ni+1], repeatability, correspCount);

        repeatabilities.emplace_back(repeatability);
        correspCounts.emplace_back(correspCount);
//        vector<vector<DMatch> > matches1to2;
//        vector<vector<uchar> > correctMatchesMask;
//        vector<vector<DMatch> > allMatches1to2;
//        vector<vector<uchar> > allCorrectMatchesMask;
//        vector<Point2f> recallPrecisionCurve;
////        Ptr<GenericDescriptorMatcher> descMatch=GenericDescriptorMatch::create("BruteForce");
////        evaluateGenericDescriptorMatcher(images[ni], images[ni+1], H,
////                                         reinterpret_cast<vector<KeyPoint> &>(ocvKeypoints[ni]),
////                                         reinterpret_cast<vector<KeyPoint> &>(ocvKeypoints[ni + 1]),
////                                         &matches1to2, &correctMatchesMask, recallPrecisionCurve,
////                                         descMatch );
//        allMatches1to2.insert( allMatches1to2.end(), matches1to2.begin(), matches1to2.end() );
//        allCorrectMatchesMask.insert( allCorrectMatchesMask.end(), correctMatchesMask.begin(), correctMatchesMask.end() );
    }
    video.release();


    return 0;
}
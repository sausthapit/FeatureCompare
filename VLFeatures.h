//
// Created by saurav on 22/07/19.
//
// Features provided by VLFeat library

#ifndef FEATURECOMPARE_VLFEATURES_H
#define FEATURECOMPARE_VLFEATURES_H

#include <opencv2/core/types.hpp>
#include "OpenCVFeatures.h"
extern "C"{
#include <vl/generic.h>
#include <vl/covdet.h>
#include <vl/liop.h>
};

class VLFeatures {

private:
    vl_size sideLength;
    VlLiopDesc * liop;
public:
    VLFeatures();
    void extractLIOP(cv::Mat image,OCVKeypoints ocvKeypoints, );


};


#endif //FEATURECOMPARE_VLFEATURES_H

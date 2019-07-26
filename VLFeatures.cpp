//
// Created by saurav on 22/07/19.
//

#include "VLFeatures.h"

VLFeatures::VLFeatures() {
    sideLength = 41 ;
    liop= vl_liopdesc_new_basic(sideLength);


}
void VLFeatures::extractLIOP(cv::Mat image, OCVKeypoints ocvKeypoints) {
    // allocate the descriptor array
    vl_size dimension = vl_liopdesc_get_dimension(liop) ;

    float *desc=new float[dimension];
//    void *desc = vl_malloc(sizeof(float) * dimension*ocvKeypoints.size()) ;

    // compute descriptor from a patch (an array of length sideLegnth *
// sideLength)
//    vl_liopdesc_process(liop, desc,image ) ;
// delete the object
//    vl_liopdesc_delete(liop) ;
    for_each(ocvKeypoints.begin(),ocvKeypoints.end(),[image](const auto &kp){

    });
}
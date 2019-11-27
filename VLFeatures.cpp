//
// Created by saurav on 22/07/19.
//

#include "VLFeatures.h"
#include <iostream>
#include <opencv2/highgui.hpp>
VLFeatures::VLFeatures() {
    sideLength = 41 ;
    liop= vl_liopdesc_new_basic(sideLength);


}
void VLFeatures::extractLIOP(const cv::Mat image,const OCVKeypoints& ocvKeypoints,OCVDescriptors& LIOPDescriptors) {

	//At the moment ignores the scale of the keypoint
    // allocate the descriptor array
    vl_size dimension = vl_liopdesc_get_dimension(liop) ;

   
//    void *desc = vl_malloc(sizeof(float) * dimension*ocvKeypoints.size()) ;

    // compute descriptor from a patch (an array of length sideLegnth *
// sideLength)
	
// delete the object
//    vl_liopdesc_delete(liop) ;
	cv::Mat imageFloat;
	image.convertTo(imageFloat, CV_32F);
	int sideLength = 41;
    for_each(ocvKeypoints.begin(),ocvKeypoints.end(),[this, imageFloat, dimension, &LIOPDescriptors,sideLength](const std::pair<std::string, std::vector<cv::KeyPoint>> & kp_vector_with_name){
		std::string det = kp_vector_with_name.first;
		std::cout <<"detector: "<< det << std::endl;
		std::vector<cv::KeyPoint> kp_vector = kp_vector_with_name.second;
		cv::Mat liopDescriptors(kp_vector.size(), dimension, CV_32F);

		//for_each(kp_vector.begin(), kp_vector.end(), [this, imageFloat, dimension,&A](const cv::KeyPoint kp) {
		int index = 0;
		for (std::vector<cv::KeyPoint>::iterator it = kp_vector.begin(); it != kp_vector.end(); ++it){
			cv::KeyPoint kp = *it;
			int half_length = cvFloor(sideLength / 2);

			cv::Rect myROI(kp.pt.x-half_length, kp.pt.y-half_length, sideLength, sideLength);
			bool is_inside = (myROI & cv::Rect(0, 0, imageFloat.cols, imageFloat.rows)) == myROI;
			if (!is_inside)
				continue;
			cv::Mat patch = imageFloat(myROI);
			//float* desc = new float[dimension];
#ifdef DEBUG
			cv::Mat img_uint;
			
			cv::normalize(img, img_uint, 0, 1, cv::NORM_MINMAX);
			cv::rectangle(img_uint, myROI, cv::Scalar(0, 255, 0))
			cv::imshow("patch", patch_uint);
			cv::waitKey(1);
#endif // DEBUG

			cv::Mat desc_row(1, dimension, CV_32F);
			vl_liopdesc_process(liop, desc_row.ptr<float>(), patch.ptr<float>());
			
			desc_row.copyTo(liopDescriptors.row(index++));
			desc_row.release();
			}
		
		std::pair<std::string, std::string> det_desc = std::pair(det, "LIOP");
		LIOPDescriptors[det_desc] = liopDescriptors;

		
		
		
		
			std::cout << "done";
    });
}
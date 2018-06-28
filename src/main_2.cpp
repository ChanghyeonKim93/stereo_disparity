#include <ros/ros.h>

#include <iostream>
#include <sys/time.h>
#include <Eigen/Dense>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ctime>

#include "StereoDisparity.h"


int main(int argc, char **argv)
{
	clock_t start, finish;
	
	ros::init(argc, argv, "stereo_logger_node");
	ros::NodeHandle nh("~");
	cv::Mat leftImg, rightImg;
	cv::Mat dispImg;
	
	char* leftWindowName      = "leftImg";
	char* rightWindowName     = "rightImg";
	char* leftMaskWindowName  = "leftmask";
	char* rightMaskWindowName = "rightmask";
	char* disparityWindowName = "disparity";

	cv::namedWindow(leftWindowName);
	cv::namedWindow(rightWindowName);
	cv::namedWindow(leftMaskWindowName);
	cv::namedWindow(rightMaskWindowName);
	cv::namedWindow(disparityWindowName);

	// ros::spinOnce(); // so fast

	leftImg  = cv::imread("/home/icslkchlap/Documents/DATASET/kitti/left/000000.png",0);
	rightImg = cv::imread("/home/icslkchlap/Documents/DATASET/kitti/right/000000.png",0);
	if(leftImg.data == NULL)
	{
		printf("ERROR- Empty input image ! \n");
		exit(1);
	}
	if(rightImg.data == NULL)
	{
		printf("ERROR- Empty input image ! \n");
		exit(1);
	}

	cv::resize(leftImg, leftImg,   cv::Size( leftImg.cols/2, leftImg.rows/2 ), 0, 0, CV_INTER_NN);
	cv::resize(rightImg, rightImg, cv::Size( rightImg.cols/2, rightImg.rows/2 ), 0, 0, CV_INTER_NN);

	cv::Mat leftMask  = cv::Mat(leftImg.rows, leftImg.cols, CV_8UC1, cv::Scalar(255));
	cv::Mat rightMask = cv::Mat(rightImg.rows, rightImg.cols, CV_8UC1, cv::Scalar(255));
 	cv::Mat rightMaskDilate = cv::Mat(rightImg.rows, rightImg.cols, CV_8UC1);
	
    start  = clock();
	StereoDisparity::calc_roi_mask(leftImg, leftMask); // 3.3 ms
	StereoDisparity::calc_roi_mask(rightImg, rightMask); // 3.3 ms
	StereoDisparity::image_dilate(rightMask, true, false, rightMaskDilate);
	StereoDisparity::calc_mask_disparity(leftImg, rightImg, leftMask, rightMaskDilate, 5, 5, 50, true, false, dispImg);
	finish = clock();


	cv::imshow(leftWindowName,leftImg);
	cv::imshow(rightWindowName,rightImg);

	cv::imshow("leftmask",  leftMask);
	cv::imshow("rightmask", rightMaskDilate);

    double duration = (double)(finish - start)/1000000.0;
    printf("실행 시간 : %f 초\n", duration);

	double min, max;
	cv::minMaxIdx(dispImg,&min,&max);
	std::cout<<min<<","<<max<<std::endl;
	cv::Mat adjMap, falseColorsMap;
	dispImg.convertTo(adjMap,CV_8UC1, 255/(max-min),-min);
	cv::applyColorMap(adjMap, falseColorsMap, cv::COLORMAP_AUTUMN);
	cv::imshow(disparityWindowName, falseColorsMap);
	cv::waitKey(0);
	printf("Done stereo_disparity.\n");

	return 0;
}

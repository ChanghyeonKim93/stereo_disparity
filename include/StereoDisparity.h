#ifndef _STEREO_DISPARITY_
#define _STEREO_DISPARITY_

#include <iostream>
#include <vector>
#include <ctime>
#include <string>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define PI 3.141592653589793238

namespace StereoDisparity {
  void calc_mask_disparity(const cv::Mat& leftImg, const cv::Mat& rightImg, const cv::Mat& leftMaskImg, const cv::Mat& rightMaskImg, const int& patchRadius, const int& minDisp, const int& maxDisp, const int& onInterp, const int& onOutlier, cv::Mat& dispImg);
  void image_dilate(const cv::Mat& inputImg, const int& onHorizontal, const int& onVertical, cv::Mat& outputImg);
  void calc_roi_mask(const cv::Mat& inputImg, cv::Mat& maskImg);
  void calc_gradient(const cv::Mat& imgInput, cv::Mat& imgGradx, cv::Mat& imgGrady, cv::Mat& imgGrad, const bool& doGaussian);
};
#endif

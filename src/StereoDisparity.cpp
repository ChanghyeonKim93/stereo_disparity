#include "StereoDisparity.h"

  void StereoDisparity::calc_mask_disparity(const cv::Mat& leftImg, const cv::Mat& rightImg, const cv::Mat& leftMaskImg, const cv::Mat& rightMaskImg, const int& patchRadius, const int& minDisp, const int& maxDisp, const int& onInterp, const int& onOutlier, cv::Mat& dispImg) {
	// input image  : leftImg (CV_8UC1, uchar), rightImg (CV_8UC1, uchar)
	// output image : dispImg (CV_64FC1, double)
	if(leftImg.type() != CV_8UC1 || rightImg.type() != CV_8UC1)
	{
		printf("ERROR - function calc_mask_disparity. input images are not CV_8UC1.\n");
		exit(1);
	}
	if(leftMaskImg.type() != CV_8UC1 || rightMaskImg.type() != CV_8UC1)
	{
		printf("ERROR - function calc_mask_disparity. input mask images are not CV_8UC1.\n");
		exit(1);	
	}
	
	int numRows = leftImg.rows, numCols = leftImg.cols;

	// initialize
	dispImg.release();
	dispImg = cv::Mat(numRows, numCols, CV_8UC1, cv::Scalar(0));

	for(int v = 0 + patchRadius; v < numRows - patchRadius; v++)
	{
		
		const uchar* leftStripData  = (const uchar*) leftImg.data;
		const uchar* rightStripData = (const uchar*) rightImg.data;

		const uchar* leftMaskData   = (const uchar*) leftMaskImg.data;
		const uchar* rightMaskData  = (const uchar*) rightMaskImg.data;
		uchar* dispImgData          = (uchar*) dispImg.data;

		for(int u = 0 + maxDisp + patchRadius; u < numCols - patchRadius; u++)
		{
			double curMin = 1e99;
			int curIdx    = 0;

			// If the current pixel of the leftImg is a interesting point, calculate the disparity.
			if( leftMaskData[v*numCols + u] > 0 )
			{
				double sumDist[numCols] = {0.0};
				for(int i = u - maxDisp; i < u - minDisp; i++)
				{
					if( rightMaskData[v*numCols + i] > 0 )
					{
						// SSD
	    				// for(int nn = -patchRadius; nn < patchRadius; nn++) sumDist[i] += (leftStripData[v*numCols + u + nn] - rightStripData[v*numCols + i + nn]) * (leftStripData[v*numCols + u + nn] - rightStripData[v*numCols + i + nn]);						
						//for(int nn = -patchRadius; nn < patchRadius; nn++) sumDist[i] += fabs(leftStripData[(v-2)*numCols + u + nn] - rightStripData[(v-2)*numCols + i + nn]);						
						for(int vv = -patchRadius; vv < patchRadius; vv++)						
						{
							for(int uu = -patchRadius; uu < patchRadius; uu++)
							{
								sumDist[i] += fabs(leftStripData[(v+vv)*numCols + u + uu] - rightStripData[(v+vv)*numCols + i + uu]);
							}
						}
						
						//std::cout<<sumDist[i]<<std::endl;
						// NCC, deprived

						if(sumDist[i] < curMin)
						{
							curMin = sumDist[i];
							curIdx = i;
						}
					}
				}

				if(onInterp == true)
				{
					double x0 = curIdx-1;
					double x1 = curIdx;
					double x2 = curIdx+1;
					double y0 = sumDist[curIdx-1];
					double y1 = sumDist[curIdx];
					double y2 = sumDist[curIdx+1];

					double x_optimal = -0.5*( y0*(x2*x2 - x1*x1) + y1*(x0*x0 - x2*x2) + y2*(x1*x1 - x0*x0) )/(y0*(x1-x2) + y1*(x2-x0) + y2*(x0-x1) );
					dispImgData[v*numCols+u] = (double)u - x_optimal;
				}
				else // do not use 
				{
					//std::cout<<u-curIdx<<std::endl;
					dispImgData[v*numCols+u] = (u - curIdx);
				}
			}
		}
	}
};


void StereoDisparity::image_dilate(const cv::Mat& inputImg, const int& onHorizontal, const int& onVertical, cv::Mat& outputImg) {
	if(inputImg.type() != CV_8UC1)
	{
		printf("ERROR- function image_dilate : input type error.\n");
	}
	const uchar* inputImgData = (const uchar*) inputImg.data;
	uchar* outputImgData      = (uchar*) outputImg.data;
	// [row*imgCol + col]
	int numRows = inputImg.rows, numCols = inputImg.cols;
	for(int v = 1; v < numRows-1; v++)
	{
		int currRows = v*numCols;
		for(int u = 1; u < numCols-1; u++)
		{
			if(inputImgData[currRows + u] == 255)
			{
				if(onHorizontal) 
				{
					outputImgData[currRows + u - 1] = 255;
					outputImgData[currRows + u]     = 255;
					outputImgData[currRows + u + 1] = 255;
				}
				if(onVertical)
				{
					outputImgData[currRows-numCols + u] = 255;
					outputImgData[currRows+numCols + u] = 255;
				}
			}
			else
			{
				outputImgData[currRows + u]     = 0;
			}
		}
	}
};

void StereoDisparity::calc_roi_mask(const cv::Mat& inputImg, cv::Mat& maskImg){

	// Parameters
	int cannyLowThres      = 163; // 0.08 of matlab
	int cannyHighThres     = 300; // 0.15 of matlab
	double gradOrientThres = sin(70.0/180.0*PI);
	double gradNormThres   = 0.0;

	// Validity test
	if(inputImg.type()!=CV_8UC1)
	{
		printf("ERROR : function calc_roi_mask : input image type error!\n");
		exit(1);
	}
	int numRows = inputImg.rows, numCols = inputImg.cols;

	// initialize the outputImg.
	maskImg.release();
	maskImg = cv::Mat(numRows, numCols, CV_8UC1, cv::Scalar(0));
	// maskImg.create(numRows, numCols, CV_8UC1);

	// [1] calculate canny edge.
	cv::Mat cannyMaskImg; 
	cv::Canny(inputImg, cannyMaskImg, cannyLowThres, cannyHighThres);

	// [2] gradient
	cv::Mat gradMaskImg = cv::Mat(numRows, numCols, CV_8UC1, cv::Scalar(0));
	cv::Mat gradxShortImg, gradyShortImg;
	cv::Mat gradImg, gradxImg, gradyImg;
	cv::Sobel(inputImg, gradxShortImg, CV_16S, 1,0,3,1,0,cv::BORDER_DEFAULT); // CV_16S : short -32768~32768, CV_64F : double
	cv::Sobel(inputImg, gradyShortImg, CV_16S, 0,1,3,1,0,cv::BORDER_DEFAULT);
	//cv::GaussianBlur(gradxShortImg, gradxShortImg, cv::Size(3,3),0.1,0.1);
	//cv::GaussianBlur(gradyShortImg, gradyShortImg, cv::Size(3,3),0.1,0.1);
	
	gradImg.create(numRows, numCols, CV_64F);
	short* gradxShortImgData = (short*) gradxShortImg.data;
	short* gradyShortImgData = (short*) gradyShortImg.data;

	uchar* gradMaskImgData   = (uchar*) gradMaskImg.data;

	for(int v = 0; v < numRows; v++){
		int currRows = v*numCols;
		for(int u = 0; u < numCols; u++){
			double gradNorm = sqrt( (double)( gradxShortImgData[currRows+u]*gradxShortImgData[currRows+u] + gradyShortImgData[currRows+u]*gradyShortImgData[currRows+u] )  );			
			if(fabs( (double)(gradyShortImgData[currRows+u])/gradNorm ) < gradOrientThres && gradNorm > gradNormThres)
			{
				gradMaskImgData[currRows + u] = 255;
			}
		}
	}

	// [3] allocation
	uchar* maskImgData           = (uchar*) maskImg.data;
	uchar* cannyMaskImgData      = (uchar*) cannyMaskImg.data;
	//uchar* gradMaskImgData       = (uchar*) gradMaskImg.data; // previously defined

	for(int v = 0; v < numRows; v++)
	{
		int curRows = v*numCols;
		for(int u = 0; u< numCols; u++)
		{
			if(cannyMaskImgData[curRows + u] == 255 && gradMaskImgData[curRows + u] == 255)
			{
				maskImgData[curRows + u] = 255;
			}
		}
	}
	
}



double norm_vector(const std::vector<double>& inputVec){
	int len = inputVec.size();
	double sum = 0.0;
	for(int i = 0; i < len; i++)
	{
		sum += inputVec[i]*inputVec[i];
	}
	return sqrt(sum);
};


double inner_vector(const std::vector<double>& inputVec1, const std::vector<double>& inputVec2){
	if(inputVec1.size() != inputVec2.size())
	{
		printf("ERROR - function cross_vector, both vector sizes are not matched.\n");
		exit(1);
	}
	int len = inputVec1.size();
	double sum = 0.0;
	for(int i = 0; i < len; i++)
	{
		sum += inputVec1[i]*inputVec2[i];
	}
	return sum;
};













void calc_gradient(const cv::Mat& imgInput, cv::Mat& imgGradx, cv::Mat& imgGrady, cv::Mat& imgGrad, const bool& doGaussian){
  // calculate gradient along each direction.
  cv::Mat imgGradxShort, imgGradyShort;
  cv::Sobel(imgInput, imgGradxShort, CV_16S, 1,0,3,1,0,cv::BORDER_DEFAULT); // CV_16S : short -32768~32768, CV_64F : double
  cv::Sobel(imgInput, imgGradyShort, CV_16S, 0,1,3,1,0,cv::BORDER_DEFAULT);

  if(doGaussian == true){ // apply Gaussian filtering or not
    cv::GaussianBlur(imgGradxShort, imgGradxShort, cv::Size(3,3),0.1	,0.1);
    cv::GaussianBlur(imgGradxShort, imgGradxShort, cv::Size(3,3),0.1,0.1);
  }
  // calculate gradient norm
  imgGrad.create(imgInput.size(), CV_64F);
  imgGradx.create(imgInput.size(), CV_64F);
  imgGrady.create(imgInput.size(), CV_64F);

  int u, v;
  for(v = 0; v < imgInput.rows; v++){
    short* imgGradxShortPtr = imgGradxShort.ptr<short>(v);
    short* imgGradyShortPtr = imgGradyShort.ptr<short>(v);
	double* imgGradxPtr     = imgGradx.ptr<double>(v);
	double* imgGradyPtr     = imgGrady.ptr<double>(v);
    double* imgGradPtr      = imgGrad.ptr<double>(v);
    for(u = 0; u < imgInput.cols; u++){
      *(imgGradPtr+u) = sqrt( (double)( ( *(imgGradxShortPtr + u) ) * ( *(imgGradxShortPtr + u) ) + ( *(imgGradyShortPtr + u) ) * ( *(imgGradyShortPtr + u) ) ) );
	  *(imgGradxPtr+u) = (double)(*(imgGradxShortPtr+u))/(*(imgGradPtr+u));
	  *(imgGradyPtr+u) = (double)(*(imgGradyShortPtr+u))/(*(imgGradPtr+u));
	}
  }
}
#include <ros/ros.h>
#include <TopicLogger.h>

#include <iostream>
#include <sys/time.h>
#include <Eigen/Dense>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// These headers are needed to subscribe stereo image pair.
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>

#define APPROXIMATE 1

#ifdef EXACT
#include <message_filters/sync_policies/exact_time.h>
#endif
#ifdef APPROXIMATE
#include <message_filters/sync_policies/approximate_time.h>
#endif

bool stereoImgUpdated = false;

typedef std::string TopicTime;

TopicTime stereoImgTime;

cv::Mat currImgLeft;
cv::Mat currImgRight;

std::string dtos(double x){
	std::stringstream s;
	s<<std::setprecision(6) << std::fixed << x;
	return s.str();
}

void stereo_image_callback(const sensor_msgs::ImageConstPtr& msgLeft , const sensor_msgs::ImageConstPtr& msgRight ) {
	cv_bridge::CvImagePtr imgLeft, imgRight;

	try{
		imgLeft = cv_bridge::toCvCopy(*msgLeft,  sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception:  %s", e.what());
		return;
	}
	try{
		imgRight = cv_bridge::toCvCopy(*msgRight, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("cv_bridge exception:  %s", e.what());
		return;
	}

	cv::Mat& matImgLeft     = imgLeft->image;
	cv::Mat& matImgRight    = imgRight->image;
	cv::Mat  matImgLeftGray, matImgRightGray;

	cv::cvtColor(matImgLeft,  currImgLeft,  CV_RGB2GRAY);
	cv::cvtColor(matImgRight, currImgRight, CV_RGB2GRAY);
	
	double currTime = (double)(msgLeft->header.stamp.sec*1e6+msgLeft->header.stamp.nsec/1000)/1000000.0;
	stereoImgTime = dtos(currTime);
	stereoImgUpdated = true;

	ROS_INFO_STREAM("Stereo images are updated.\n");
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "stereo_logger_node");
	ros::NodeHandle nh("~");

    std::string image_left_name = std::string("/stereo/left/image_raw");
	std::string image_right_name = std::string("/stereo/right/image_raw");
    ros::param::get("~image_left_topic_name", image_left_name);
	ros::param::get("~image_right_topic_name", image_right_name);

	std::string mydir;
	if(ros::param::get("~folder_dir", mydir)==false){

		ROS_ERROR_STREAM("This is not an existing folder ! \n");
	}
	std::cout<<"Save folder directory : " <<mydir<<std::endl;
	TopicLogger *topic_logger = new TopicLogger(mydir);

	message_filters::Subscriber<sensor_msgs::Image> leftImgSubscriber(nh , image_left_name , 1 );
	message_filters::Subscriber<sensor_msgs::Image> rightImgSubscriber(nh , image_right_name , 1 );

	#ifdef EXACT
		typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
	#endif
	#ifdef APPROXIMATE
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> MySyncPolicy;
	#endif

	 message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(4), leftImgSubscriber, rightImgSubscriber );
	sync.registerCallback(boost::bind(&stereo_image_callback, _1, _2));


	while(ros::ok()){
		ros::spinOnce(); // so fast
		if(stereoImgUpdated==true){
			topic_logger->stereo_img_addline(currImgLeft, currImgRight, stereoImgTime);
		}
	}
	delete topic_logger;
	return 0;
}

//
// Created by ksals on 2022/7/14.
//

#pragma once

#include <ros/ros.h>
#include <ros/package.h>
#include <opencv2/dnn.hpp>
#include <nodelet/nodelet.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <image_transport/camera_subscriber.h>
#include <vector>
#include <string>

class FaceDetection : public nodelet::Nodelet
{
private:
  ros::NodeHandle parent_nh_;
  std::shared_ptr<image_transport::ImageTransport> it_;
  image_transport::CameraSubscriber cam_sub_;
  cv::Mat_<cv::Vec3b> raw_image_;
  sensor_msgs::CameraInfoConstPtr cam_info_;
  image_transport::Publisher image_pub_;
  std::string package_path_;
  std::string model_path_;

  void onFrameCb(const sensor_msgs::ImageConstPtr& img, const sensor_msgs::CameraInfoConstPtr& info);

public:
  FaceDetection() = default;

  ~FaceDetection() = default;

  void initialize(ros::NodeHandle &nh);

  void detectFace(cv_bridge::CvImagePtr& cv_image);

  void publishImage();

  void onInit() override;

  template<typename T>
  inline T getParam(const ros::NodeHandle &nh, const std::string &param_name, const T &default_val)
  {
    T param_val;
    nh.param<T>(param_name, param_val, default_val);
    return param_val;
  }
};

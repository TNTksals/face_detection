//
// Created by ksals on 2022/7/14.
//

#include "face_detection/face_detection.h"
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(FaceDetection, nodelet::Nodelet);

void FaceDetection::onInit()
{
  ros::NodeHandle &nh = getPrivateNodeHandle();
  initialize(nh);
}

void FaceDetection::initialize(ros::NodeHandle &nh)
{
  parent_nh_ = ros::NodeHandle(nh, "face_detection");
  it_ = std::make_shared<image_transport::ImageTransport>(parent_nh_);

  package_path_ = ros::package::getPath("face_detection");
  if (package_path_.empty())
    ROS_INFO("Fail to find the face_detection package.");
  else
    ROS_INFO("Package path: %s", package_path_.c_str());
  model_path_ = package_path_ + "/model/";

  cam_sub_ = it_->subscribeCamera("/hk_camera/image_raw", 1, &FaceDetection::onFrameCb, this);
  image_pub_ = it_->advertise("/face_detection/output_image", 1);
  ROS_INFO("Success.");
}

void FaceDetection::onFrameCb(const sensor_msgs::ImageConstPtr &img, const sensor_msgs::CameraInfoConstPtr &info)
{
  cam_info_ = info;
  auto cv_image_ptr = boost::const_pointer_cast<cv_bridge::CvImage>(cv_bridge::toCvShare(img, "bgr8"));
  ROS_ASSERT(cv_image_ptr != nullptr);
  detectFace(cv_image_ptr);
  publishImage();
}

void FaceDetection::detectFace(cv_bridge::CvImagePtr& cv_image)
{
  cv_image->image.copyTo(raw_image_);
  cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_path_ + "opencv_face_detector_uint8.pb",
                                                    model_path_ + "opencv_face_detector.pbtxt");
  cv::Mat blob = cv::dnn::blobFromImage(raw_image_, 1.0, cv::Size(300, 300),
                                        cv::Scalar(104, 177, 123), false, false);
  net.setInput(blob);
  cv::Mat probs = net.forward();
  cv::Mat_<float> result = cv::Mat(probs.size[2], probs.size[3], CV_32F, probs.ptr<float>());
  for (int i = 0; i < result.rows; i++)
  {
    float confidence = result(i, 2);
    if (confidence > 0.5)
    {
      int x1 = static_cast<int>(result(i, 3) * raw_image_.cols);
      int y1 = static_cast<int>(result(i, 4) * raw_image_.rows);
      int x2 = static_cast<int>(result(i, 5) * raw_image_.cols);
      int y2 = static_cast<int>(result(i, 6) * raw_image_.rows);
      cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
      cv::rectangle(raw_image_, roi, cv::Scalar(0, 255, 0), 2, 8, 0);
    }
  }
}

void FaceDetection::publishImage()
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", raw_image_).toImageMsg();
  image_pub_.publish(msg);
}

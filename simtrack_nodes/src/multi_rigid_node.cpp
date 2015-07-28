/*****************************************************************************/
/*  Copyright (c) 2015, Karl Pauwels                                         */
/*  All rights reserved.                                                     */
/*                                                                           */
/*  Redistribution and use in source and binary forms, with or without       */
/*  modification, are permitted provided that the following conditions       */
/*  are met:                                                                 */
/*                                                                           */
/*  1. Redistributions of source code must retain the above copyright        */
/*  notice, this list of conditions and the following disclaimer.            */
/*                                                                           */
/*  2. Redistributions in binary form must reproduce the above copyright     */
/*  notice, this list of conditions and the following disclaimer in the      */
/*  documentation and/or other materials provided with the distribution.     */
/*                                                                           */
/*  3. Neither the name of the copyright holder nor the names of its         */
/*  contributors may be used to endorse or promote products derived from     */
/*  this software without specific prior written permission.                 */
/*                                                                           */
/*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      */
/*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        */
/*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    */
/*  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     */
/*  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   */
/*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         */
/*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    */
/*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    */
/*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      */
/*  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    */
/*  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     */
/*****************************************************************************/

#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <simtrack_nodes/multi_rigid_node.h>
#include <tf/transform_datatypes.h>
#include <windowless_gl_context.h>
#undef Success
#include <Eigen/Geometry>
#include <translation_rotation_3d.h>
#include <hdf5_file.h>
#include <utilities.h>

using namespace util;

namespace simtrack {

void MultiRigidNode::detectorThreadFunction(cv::Mat camera_matrix, size_t width,
                                            size_t height) {

  // initialize CUDA in detector thread
  util::initializeCUDARuntime(device_id_detector_);

  int detector_object_index = 0;

  multi_rigid_detector_ =
      interface::MultiRigidDetector::Ptr(new interface::MultiRigidDetector(
          width, height, camera_matrix, obj_filenames_, device_id_detector_,
          parameters_detector_));

  while (!shutdown_detector_.load()) {

    if (detector_enabled_.load()) {

      // update selected objects if new objects selected
      if (switched_detector_objects_.load()) {
        {
          std::lock_guard<std::mutex> lock(obj_filenames_mutex_);
          multi_rigid_detector_->setObjects(obj_filenames_);
        }
        detector_object_index = 0;
        switched_detector_objects_.store(false);
      }

      // estimate pose if objects loaded in detector
      if (multi_rigid_detector_->getNumberOfObjects() > 0) {

        // process frame
        cv::Mat img_gray;
        {
          std::lock_guard<std::mutex> lock(img_gray_detector_mutex_);
          img_gray = img_gray_detector_.clone();
        }
        pose::TranslationRotation3D detector_pose;
        multi_rigid_detector_->estimatePose(img_gray, detector_object_index,
                                            detector_pose);

        // transmit pose to tracker
        {
          std::lock_guard<std::mutex> lock(most_recent_detector_pose_mutex_);
          most_recent_detector_object_index_ = detector_object_index;
          most_recent_detector_pose_ = detector_pose;
        }

        // select next object to detect
        detector_object_index = (detector_object_index + 1) %
                                multi_rigid_detector_->getNumberOfObjects();

      } else {
        // reduce load on this thread when no objects selected
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
    }
  }
}

MultiRigidNode::MultiRigidNode(ros::NodeHandle nh)
    : nh_(nh), device_id_detector_(0), most_recent_detector_object_index_(0),
      detector_enabled_(true), shutdown_detector_(false), ready_(false),
      output_image_(
          interface::MultiRigidTracker::OutputImageType::model_appearance),
      recording_(false), root_recording_path_("/dev/shm/"), frame_count_(0),
      recording_start_time_(ros::Time::now()), auto_disable_detector_(false),
      color_only_mode_(false), switched_tracker_objects_(false) {
  // get model names from parameter server
  if (!ros::param::get("/simtrack/model_path", model_path_))
    throw std::runtime_error(
        std::string("MultiRigidNode::MultiRigidNode: could not "
                    "find /simtrack/model_path on parameter server\n"));

  std::vector<std::string> model_names;
  if (!ros::param::get("/simtrack/model_names", model_names))
    throw std::runtime_error(
        std::string("MultiRigidNode::MultiRigidNode: could not "
                    "find /simtrack/model_names on parameter server\n"));

  for (auto &it : model_names) {
    objects_.push_back(composeObjectInfo(it));
    obj_filenames_.push_back(composeObjectFilename(it));
    pose_publishers_[it] = nh.advertise<geometry_msgs::PoseStamped>("/simtrack/"+it, 1);;
  }

  // get optical flow parameters
  ros::param::get("simtrack/optical_flow/n_scales", parameters_flow_.n_scales_);
  ros::param::get("simtrack/optical_flow/median_filter",
                  parameters_flow_.median_filter_);
  ros::param::get("simtrack/optical_flow/consistent",
                  parameters_flow_.consistent_);
  ros::param::get("simtrack/optical_flow/cons_thres",
                  parameters_flow_.cons_thres_);
  ros::param::get("simtrack/optical_flow/four_orientations",
                  parameters_flow_.four_orientations_);

  // get pose tracker parameters
  ros::param::get("simtrack/tracker/color_only_mode", color_only_mode_);
  ros::param::get("simtrack/tracker/n_icp_outer_it",
                  parameters_pose_.n_icp_outer_it_);
  ros::param::get("simtrack/tracker/n_icp_inner_it",
                  parameters_pose_.n_icp_inner_it_);
  ros::param::get("simtrack/tracker/w_flow", parameters_pose_.w_flow_);
  ros::param::get("simtrack/tracker/w_ar_flow", parameters_pose_.w_ar_flow_);
  ros::param::get("simtrack/tracker/w_disp", parameters_pose_.w_disp_);
  ros::param::get("simtrack/tracker/max_samples",
                  parameters_pose_.max_samples_);
  int key_bits = parameters_pose_.getKeyBits();
  ros::param::get("simtrack/tracker/key_bits", key_bits);
  parameters_pose_.setKeyBits(key_bits);
  ros::param::get("simtrack/tracker/near_plane", parameters_pose_.near_plane_);
  ros::param::get("simtrack/tracker/far_plane", parameters_pose_.far_plane_);
  ros::param::get("simtrack/tracker/reliability_threshold",
                  parameters_pose_.reliability_threshold_);
  ros::param::get("simtrack/tracker/max_proportion_projected_bounding_box",
                  parameters_pose_.max_proportion_projected_bounding_box_);
  ros::param::get("simtrack/tracker/sparse_intro_reliability_threshold",
                  parameters_pose_.sparse_intro_reliability_threshold_);
  ros::param::get("simtrack/tracker/sparse_intro_allowed_reliability_decrease",
                  parameters_pose_.sparse_intro_allowed_reliability_decrease_);
  ros::param::get("simtrack/tracker/max_t_update_norm_squared",
                  parameters_pose_.max_t_update_norm_squared_);

  // get detector parameters
  ros::param::get("simtrack/detector/device_id", device_id_detector_);
  ros::param::get("simtrack/detector/vec_size", parameters_detector_.vec_size_);
  ros::param::get("simtrack/detector/num_iter_ransac",
                  parameters_detector_.num_iter_ransac_);

  /*****************************/
  /* Setup CUDA for GL interop */
  /*****************************/

  int device_id_tracker = 0;
  ros::param::get("simtrack/tracker/device_id", device_id_tracker);

  // Create dummy GL context before cudaGL init
  render::WindowLessGLContext dummy(10, 10);

  // CUDA Init
  util::initializeCUDARuntime(device_id_tracker);

  // auto-disable detector in case of single gpu
  auto_disable_detector_ = (device_id_tracker == device_id_detector_);

  ready_ = true;
}

MultiRigidNode::~MultiRigidNode() {
  // cleanly shutdown detector thread (if running)
  if (detector_thread_ != nullptr) {
    shutdown_detector_.store(true);
    detector_thread_->join();
  }
}

bool MultiRigidNode::start() {
  if (!ready_) {
    return false;
  }

  switch_objects_srv_ = nh_.advertiseService(
      "/simtrack/switch_objects", &MultiRigidNode::switchObjects, this);

  bool compressed_streams = false;
  ros::param::get("simtrack/use_compressed_streams", compressed_streams);

  image_transport::TransportHints rgb_hint, depth_hint;
  if (compressed_streams) {
    rgb_hint = image_transport::TransportHints("compressed");
    depth_hint = image_transport::TransportHints("compressedDepth");
  } else {
    rgb_hint = image_transport::TransportHints("raw");
    depth_hint = image_transport::TransportHints("raw");
  }

  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_rgb_.subscribe(*rgb_it_, "rgb", 1, rgb_hint);
  sub_rgb_info_.subscribe(nh_, "rgb_info", 1);

  if (color_only_mode_) {
    sync_rgb_.reset(
        new SynchronizerRGB(SyncPolicyRGB(5), sub_rgb_, sub_rgb_info_));
    sync_rgb_->registerCallback(
        boost::bind(&MultiRigidNode::colorOnlyCb, this, _1, _2));
  } else {
    depth_it_.reset(new image_transport::ImageTransport(nh_));
    sub_depth_.subscribe(*depth_it_, "depth", 1, depth_hint);
    sync_rgbd_.reset(new SynchronizerRGBD(SyncPolicyRGBD(5), sub_depth_,
                                          sub_rgb_, sub_rgb_info_));
    sync_rgbd_->registerCallback(
        boost::bind(&MultiRigidNode::depthAndColorCb, this, _1, _2, _3));
  }

  debug_img_it_.reset(new image_transport::ImageTransport(nh_));
  debug_img_pub_ = debug_img_it_->advertise("/simtrack/image", 1);

  dynamic_reconfigure::Server<simtrack_nodes::VisualizationConfig>::CallbackType
  f;
  f = boost::bind(&MultiRigidNode::reconfigureCb, this, _1, _2);
  dynamic_reconfigure_server_.setCallback(f);

  return true;
}

bool MultiRigidNode::switchObjects(simtrack_nodes::SwitchObjectsRequest &req,
                                   simtrack_nodes::SwitchObjectsResponse &res) {
  std::stringstream ss;
  ss << "simtrack switching to models: ";
  for (auto &it : req.model_names)
    ss << it << " ";
  ROS_INFO("%s", ss.str().c_str());

  // switch tracker
  objects_.clear();
  for (auto &it : req.model_names)
    objects_.push_back(composeObjectInfo(it));
  switched_tracker_objects_ = true;

  // switch detector
  // the detector may start issuing poses out of bounds to the tracker
  {
    std::lock_guard<std::mutex> lock(obj_filenames_mutex_);
    obj_filenames_.clear();
    for (auto &it : req.model_names)
      obj_filenames_.push_back(composeObjectFilename(it));
  }
  switched_detector_objects_.store(true);

  return true;
}

void MultiRigidNode::depthAndColorCb(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &rgb_info_msg) {
  // we'll assume registration is correct so that rgb and depth camera matrices
  // are equal
  camera_matrix_rgb_ = composeCameraMatrix(rgb_info_msg);

  cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr;
  try {
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
    cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
  }
  catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  updatePose(cv_rgb_ptr, cv_depth_ptr, rgb_msg->header.frame_id);
}

void MultiRigidNode::colorOnlyCb(
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &rgb_info_msg) {
  // we'll assume registration is correct so that rgb and depth camera matrices
  // are equal
  camera_matrix_rgb_ = composeCameraMatrix(rgb_info_msg);

  cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr;
  try {
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
  }
  catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  updatePose(cv_rgb_ptr, cv_depth_ptr, rgb_msg->header.frame_id);
}

void MultiRigidNode::updatePose(const cv_bridge::CvImagePtr &cv_rgb_ptr,
                                const cv_bridge::CvImagePtr &cv_depth_ptr,
                                const std::string &frame_id) {
  if ((!color_only_mode_) && (cv_depth_ptr == nullptr))
    throw std::runtime_error("MultiRigidNode::updatePose: received "
                             "nullptr depth while not in color_only_mode_\n");

  // convert image to gray if required
  cv::Mat img_gray;
  if (cv_rgb_ptr->image.type() == CV_8UC3) {
    cv::cvtColor(cv_rgb_ptr->image, img_gray, CV_BGR2GRAY);
  } else if (cv_rgb_ptr->image.type() == CV_8UC1) {
    img_gray = cv_rgb_ptr->image.clone();
  } else {
    throw std::runtime_error("MultiRigidNode::updatePose: image type "
                             "should be CV_8UC3 or CV_8UC1\n");
  }

  // initialize detector thread if not yet active
  // the engine is created here since we need camera info
  if (detector_thread_ == nullptr) {
    size_t width = img_gray.cols;
    size_t height = img_gray.rows;
    detector_thread_ = std::unique_ptr<std::thread>(
        new std::thread(&MultiRigidNode::detectorThreadFunction, this,
                        camera_matrix_rgb_, width, height));
  }

  // copy the image for the detector (if running)
  if (detector_enabled_.load()) {
    std::lock_guard<std::mutex> lock(img_gray_detector_mutex_);
    img_gray_detector_ = img_gray.clone();
  }

  // initialize tracker engine if not yet active
  // the engine is created here since we need camera info
  if (multi_rigid_tracker_ == nullptr) {
    multi_rigid_tracker_ =
        interface::MultiRigidTracker::Ptr(new interface::MultiRigidTracker(
            img_gray.cols, img_gray.rows, camera_matrix_rgb_, objects_,
            parameters_flow_, parameters_pose_));
  }

  // update selected objects if new objects selected
  if (switched_tracker_objects_) {
    multi_rigid_tracker_->setObjects(objects_);
    switched_tracker_objects_ = false;
  }

  // process frame if objects loaded in tracker
  // ------------------------------------------
  if (multi_rigid_tracker_->getNumberOfObjects() > 0) {

    // update detector pose in tracker
    {
      std::lock_guard<std::mutex> lock(most_recent_detector_pose_mutex_);
      if (most_recent_detector_object_index_ <
          multi_rigid_tracker_->getNumberOfObjects()) {
        multi_rigid_tracker_->setRigidDetectorPose(
            most_recent_detector_pose_, most_recent_detector_object_index_);
      }
    }

    // update tracker pose
    if (color_only_mode_)
      multi_rigid_tracker_->updatePoses(img_gray);
    else
      multi_rigid_tracker_->updatePoses(img_gray, cv_depth_ptr->image);

    // publish reliable poses
    std::vector<geometry_msgs::Pose> poses =
        multi_rigid_tracker_->getPoseMessages();

    for (int object_index = 0; object_index < poses.size(); object_index++) {
      if (multi_rigid_tracker_->isPoseReliable(object_index)) {
        geometry_msgs::Pose curr_pose = poses.at(object_index);
        tf::StampedTransform object_transform;
        object_transform.setOrigin(tf::Vector3(
            curr_pose.position.x, curr_pose.position.y, curr_pose.position.z));
        object_transform.setRotation(
            tf::Quaternion(curr_pose.orientation.x, curr_pose.orientation.y,
                           curr_pose.orientation.z, curr_pose.orientation.w));
        //      object_transform.stamp_ = ros::Time::now();
        object_transform.stamp_ = cv_rgb_ptr->header.stamp;
        object_transform.frame_id_ = frame_id;
        object_transform.child_frame_id_ = objects_.at(object_index).label_;
        tfb_.sendTransform(object_transform);

        geometry_msgs::PoseStamped curr_pose_stamped;
        curr_pose_stamped.pose = curr_pose;
        curr_pose_stamped.header.frame_id = frame_id;
        curr_pose_stamped.header.stamp = cv_rgb_ptr->header.stamp;
        pose_publishers_[objects_.at(object_index).label_].publish(curr_pose_stamped);
      }
    }

    // disable detector if all objects tracked and auto-disable enabled
    // (single-gpu case)
    if (auto_disable_detector_)
      detector_enabled_.store(!multi_rigid_tracker_->areAllPosesReliable());

    // generate output image
    cv::Mat texture = multi_rigid_tracker_->generateOutputImage(output_image_);
    cv::Mat tmp_img = texture.clone();
    bool show_bounding_boxes = false;
    if (show_bounding_boxes) {
      auto bounding_boxes =
          multi_rigid_tracker_->getBoundingBoxesInCameraImage();
      for (int object_index = 0; object_index < poses.size(); object_index++) {
        if (multi_rigid_tracker_->isPoseReliable(object_index)) {
          for (int r = 0; r < 8; r++) {
            // draw in image
            auto p = cv::Point(bounding_boxes.at(object_index).at(r * 2),
                               bounding_boxes.at(object_index).at(r * 2 + 1));
            cv::circle(tmp_img, p, 3, CV_RGB(255, 0, 0), -1, 8);
          }
        }
      }
    }
    cv_bridge::CvImage cv_image;
    cv_image.image = tmp_img;
    cv_image.encoding = "rgba8";
    sensor_msgs::Image ros_image;
    cv_image.toImageMsg(ros_image);
    debug_img_pub_.publish(ros_image);

    // record data to new file if requested
    if (recording_) {
      // create file
      std::stringstream file_name;
      file_name << "frame_" << std::setw(6) << std::setfill('0') << frame_count_
                << ".h5";
      boost::filesystem::path file_path = recording_path_ / file_name.str();
      util::HDF5File file(file_path.string());

      // store frame time
      std::vector<int> time_size{ 1 };
      std::vector<double> time_data{(ros::Time::now() - recording_start_time_)
                                        .toSec() };
      file.writeArray("time", time_data, time_size);

      // store poses
      if (recording_flags_.poses_)
        multi_rigid_tracker_->savePoses(file);

      // store rgb
      if (recording_flags_.image_) {
        cv::Mat rgb;
        cv::cvtColor(cv_rgb_ptr->image, rgb, CV_BGR2RGB);
        std::vector<int> rgb_size = { rgb.rows, rgb.cols, rgb.channels() };
        int rgb_n = accumulate(rgb_size.begin(), rgb_size.end(), 1,
                               std::multiplies<int>());
        std::vector<uint8_t> rgb_data((uint8_t *)rgb.data,
                                      (uint8_t *)rgb.data + rgb_n);
        file.writeArray("image", rgb_data, rgb_size);
      }

      // store depth
      if (recording_flags_.depth_ && (!color_only_mode_)) {
        cv::Mat depth = cv_depth_ptr->image;
        std::vector<int> depth_size = { depth.rows, depth.cols };
        int depth_n = depth.rows * depth.cols;
        if (depth.type() == CV_32FC1) {
          std::vector<float> depth_data((float *)depth.data,
                                        (float *)depth.data + depth_n);
          file.writeArray("depth", depth_data, depth_size);
        } else if (depth.type() == CV_16UC1) {
          std::vector<uint16_t> depth_data((uint16_t *)depth.data,
                                           (uint16_t *)depth.data + depth_n);
          file.writeArray("depth", depth_data, depth_size);
        }
      }

      // store optical flow
      if (recording_flags_.optical_flow_)
        multi_rigid_tracker_->saveOpticalFlow(file);
    }
  }
  frame_count_++;
}

void MultiRigidNode::reconfigureCb(simtrack_nodes::VisualizationConfig &config,
                                   uint32_t level) {
  switch (config.visualization) {
  case 0:
    output_image_ =
        interface::MultiRigidTracker::OutputImageType::model_appearance;
    break;
  case 1:
    output_image_ =
        interface::MultiRigidTracker::OutputImageType::model_appearance_blended;
    break;
  case 2:
    output_image_ =
        interface::MultiRigidTracker::OutputImageType::optical_flow_x;
    break;
  case 3:
    output_image_ =
        interface::MultiRigidTracker::OutputImageType::optical_flow_y;
    break;
  }

  // update recording flags
  recording_flags_.poses_ = config.save_object_poses;
  recording_flags_.image_ = config.save_image;
  recording_flags_.depth_ = config.save_depth;
  recording_flags_.optical_flow_ = config.save_optical_flow;

  // generate a new folder whenever recording is activated
  if ((!recording_) && (config.start_stop_recording)) {
    // create new recording folder
    // count up from simtrack_000 until one doesn't exist
    int folder_count = 0;
    bool path_exists = true;
    while (path_exists) {
      std::stringstream relative_path;
      relative_path << "simtrack_recording_" << std::setw(3)
                    << std::setfill('0') << folder_count;
      recording_path_ = root_recording_path_ / relative_path.str();
      path_exists = boost::filesystem::exists(recording_path_);
      folder_count++;
    }
    boost::filesystem::create_directory(recording_path_);

    // save configuration file (camera_matrix, object info)
    util::HDF5File file((recording_path_ / "scene_info.h5").string());
    std::vector<int> size{ camera_matrix_rgb_.rows, camera_matrix_rgb_.cols };
    int n = accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
    std::vector<double> data((double *)camera_matrix_rgb_.data,
                             (double *)camera_matrix_rgb_.data + n);
    file.writeArray("camera_matrix_rgb", data, size);
    std::vector<std::string> object_labels, object_filenames;
    for (auto &it : objects_) {
      object_labels.push_back(it.label_);
      object_filenames.push_back(it.filename_);
    }
    size = {(int)objects_.size() };
    file.writeArray("object_labels", object_labels, size);
    file.writeArray("object_filenames", object_filenames, size);
    frame_count_ = 0;
    recording_start_time_ = ros::Time::now();
  }

  recording_ = config.start_stop_recording;
}

interface::MultiRigidTracker::ObjectInfo
MultiRigidNode::composeObjectInfo(std::string model_name) {
  std::string obj_file_name =
      model_path_ + "/" + model_name + "/" + model_name + ".obj";
  return (interface::MultiRigidTracker::ObjectInfo(model_name, obj_file_name));
}

std::string MultiRigidNode::composeObjectFilename(std::string model_name) {
  return (model_path_ + "/" + model_name + "/" + model_name + "_SIFT.h5");
}

cv::Mat MultiRigidNode::composeCameraMatrix(
    const sensor_msgs::CameraInfoConstPtr &info_msg) {
  cv::Mat camera_matrix =
      cv::Mat(3, 4, CV_64F, (void *)info_msg->P.data()).clone();
  camera_matrix.at<double>(0, 2) -= info_msg->roi.x_offset;
  camera_matrix.at<double>(1, 2) -= info_msg->roi.y_offset;
  return (camera_matrix);
}

} // end namespace simtrack

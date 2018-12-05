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

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <simtrack_nodes/pr2_cam_switch_node.h>
#include <windowless_gl_context.h>
#undef Success
#include <Eigen/Geometry>
#include <translation_rotation_3d.h>
#include <hdf5_file.h>
#include <utilities.h>
#include <numeric>

using namespace util;

namespace simtrack {

void PR2CamSwitchNode::detectorThreadFunction(size_t width, size_t height) {

  cv::Mat camera_matrix = camera_matrix_rgb_;

  // initialize CUDA in detector thread
  util::initializeCUDARuntime(device_id_detector_);

  int detector_object_index = 0;

  multi_rigid_detector_ =
      interface::MultiRigidDetector::Ptr(new interface::MultiRigidDetector(
          width, height, camera_matrix, obj_filenames_, device_id_detector_,
          parameters_detector_));

  while (!shutdown_detector_.load()) {

    if (detector_enabled_.load()) {

      // update camera and pose parameters if camera switched
      // we assume image resolution doesn't change, this will trigger an
      // exception in the detector
      if (switched_detector_camera_.load()) {
        {
          std::lock_guard<std::mutex> lock(camera_matrix_rgb_mutex_);
          camera_matrix = camera_matrix_rgb_;
        }
        multi_rigid_detector_->setCameraMatrix(camera_matrix);
        switched_detector_camera_.store(false);
      }

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

        cv::Mat img_gray;
        {
          std::lock_guard<std::mutex> lock(img_gray_detector_mutex_);
          img_gray = img_gray_detector_.clone();
        }
        pose::TranslationRotation3D detector_pose;
        multi_rigid_detector_->estimatePose(img_gray, detector_object_index,
                                            detector_pose);

        // ensure all the values are finite (pnp produces nans on failure)
        detector_pose.setValid(detector_pose.isFinite());

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

PR2CamSwitchNode::PR2CamSwitchNode(ros::NodeHandle nh)
    : nh_(nh), device_id_detector_(0), most_recent_detector_object_index_(0),
      detector_enabled_(true), shutdown_detector_(false), ready_(false),
      output_image_(interface::MultiRigidTracker::OutputImageType::
                        model_appearance_blended),
      recording_(false), root_recording_path_("/dev/shm/"), frame_count_(0),
      recording_start_time_(ros::Time::now()), auto_disable_detector_(false),
      color_only_mode_(false), switched_tracker_camera_(false),
      switched_tracker_objects_(false) {
  // get model names from parameter server
  if (!ros::param::get("/simtrack/model_path", model_path_))
    parameterError(__func__, "/simtrack/model_path");

  std::vector<std::string> model_names;

  if (ros::param::get("/simtrack/model_names", model_names)) {
    for (auto &it : model_names) {
      objects_.push_back(composeObjectInfo(it));
      obj_filenames_.push_back(composeObjectFilename(it));
    }
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
  // color_only_mode defined through pr2_camera_topic.yaml
  //  ros::param::get("simtrack/tracker/color_only_mode", color_only_mode_);
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
  ros::param::get("/simtrack/tracker/device_id", device_id_tracker);

  // Create dummy GL context before cudaGL init
  render::WindowLessGLContext dummy(10, 10);

  // CUDA Init
  util::initializeCUDARuntime(device_id_tracker);

  // auto-disable detector in case of single gpu
  auto_disable_detector_ = (device_id_tracker == device_id_detector_);

  ready_ = true;
}

PR2CamSwitchNode::~PR2CamSwitchNode() {
  // cleanly shutdown detector thread (if running)
  if (detector_thread_ != nullptr) {
    shutdown_detector_.store(true);
    detector_thread_->join();
  }
}

bool PR2CamSwitchNode::start() {
  if (!ready_) {
    return false;
  }

  switch_camera_srv_ = nh_.advertiseService(
      "/simtrack/switch_camera", &PR2CamSwitchNode::switchCamera, this);

  switch_objects_srv_ = nh_.advertiseService(
      "/simtrack/switch_objects", &PR2CamSwitchNode::switchObjects, this);

  sub_joint_state_ =
      nh_.subscribe("joint_states", 1, &PR2CamSwitchNode::jointStateCb, this);

  debug_img_it_.reset(new image_transport::ImageTransport(nh_));
  debug_img_pub_ = debug_img_it_->advertise("/simtrack/image", 1);

  dynamic_reconfigure::Server<simtrack_nodes::VisualizationConfig>::CallbackType
  f;
  f = boost::bind(&PR2CamSwitchNode::reconfigureCb, this, _1, _2);
  dynamic_reconfigure_server_.setCallback(f);

  setupCameraSubscribers(0);

  return true;
}

bool PR2CamSwitchNode::switchCamera(simtrack_nodes::SwitchCameraRequest &req,
                                    simtrack_nodes::SwitchCameraResponse &res) {
  ROS_INFO("simtrack switching to camera: %d", req.camera);
  setupCameraSubscribers(req.camera);
  switched_tracker_camera_ = true;
  return true;
}

bool
PR2CamSwitchNode::switchObjects(simtrack_nodes::SwitchObjectsRequest &req,
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

void PR2CamSwitchNode::depthAndColorCb(
    const sensor_msgs::ImageConstPtr &depth_msg,
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &rgb_info_msg) {
  // we'll assume registration is correct so that rgb and depth camera matrices
  // are equal
  {
    std::lock_guard<std::mutex> lock(camera_matrix_rgb_mutex_);
    camera_matrix_rgb_ =
        cv::Mat(3, 4, CV_64F, (void *)rgb_info_msg->P.data()).clone();
  }

  cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr;
  try {
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
    cv_depth_ptr = cv_bridge::toCvCopy(depth_msg, depth_msg->encoding);
  }
  catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  updatePose(cv_rgb_ptr, cv_depth_ptr);
}

void PR2CamSwitchNode::colorOnlyCb(
    const sensor_msgs::ImageConstPtr &rgb_msg,
    const sensor_msgs::CameraInfoConstPtr &rgb_info_msg) {
  // we'll assume registration is correct so that rgb and depth camera matrices
  // are equal
  {
    std::lock_guard<std::mutex> lock(camera_matrix_rgb_mutex_);
    camera_matrix_rgb_ =
        cv::Mat(3, 4, CV_64F, (void *)rgb_info_msg->P.data()).clone();
  }

  cv_bridge::CvImagePtr cv_rgb_ptr, cv_depth_ptr;
  try {
    cv_rgb_ptr = cv_bridge::toCvCopy(rgb_msg, "bgr8");
  }
  catch (cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  updatePose(cv_rgb_ptr, cv_depth_ptr);
}

void PR2CamSwitchNode::updatePose(const cv_bridge::CvImagePtr &cv_rgb_ptr,
                                  const cv_bridge::CvImagePtr &cv_depth_ptr) {
  if ((!color_only_mode_) && (cv_depth_ptr == nullptr))
    throw std::runtime_error("PR2CamSwitchNode::updatePose: received "
                             "nullptr depth while not in color_only_mode_\n");

  // convert image to gray if required
  cv::Mat img_gray;
  if (cv_rgb_ptr->image.type() == CV_8UC3) {
    cv::cvtColor(cv_rgb_ptr->image, img_gray, CV_BGR2GRAY);
  } else if (cv_rgb_ptr->image.type() == CV_8UC1) {
    img_gray = cv_rgb_ptr->image.clone();
  } else {
    throw std::runtime_error("PR2CamSwitchNode::updatePose: image type "
                             "should be CV_8UC3 or CV_8UC1\n");
  }

  // initialize detector thread if not yet active
  // the engine is created here since we need camera info
  if (detector_thread_ == nullptr) {
    size_t width = img_gray.cols;
    size_t height = img_gray.rows;
    detector_thread_ = std::unique_ptr<std::thread>(new std::thread(
        &PR2CamSwitchNode::detectorThreadFunction, this, width, height));
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

    // create the robot object
    urdf::Model robot_model;
    robot_model.initParam("robot_description");
    robot_ = std::unique_ptr<render::Robot>{ new render::Robot(
        robot_model, multi_rigid_tracker_->getSceneManager()) };
    // make robot object invisible to tracker
    robot_->setFixedSegmentLabels(0);
  }

  // update camera and pose parameters if camera switched
  // we assume image resolution doesn't change, this will trigger an exception
  // in the tracker
  if (switched_tracker_camera_) {
    multi_rigid_tracker_->setCameraMatrix(camera_matrix_rgb_);
    // we should reset the weights since in a color-only situation
    // w_disp is automatically set to 0
    multi_rigid_tracker_->setWeights(parameters_pose_.w_flow_,
                                     parameters_pose_.w_ar_flow_,
                                     parameters_pose_.w_disp_);
    switched_tracker_camera_ = false;
    // only now allow switching detector camera since camera_matrix_rgb_ is
    // updated
    switched_detector_camera_.store(true);
  }

  // update selected objects if new objects selected
  if (switched_tracker_objects_) {
    multi_rigid_tracker_->setObjects(objects_);
    switched_tracker_objects_ = false;
  }

  // update the robot pose
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);

    if (!joint_state_.empty())
      robot_->setJointState(joint_state_);
  }

  Ogre::Vector3 camera_position;
  Ogre::Quaternion camera_orientation;
  robot_->getFrame(robot_camera_frame_id_, camera_position, camera_orientation);
  pose::TranslationRotation3D camera_frame(camera_position, camera_orientation);
  multi_rigid_tracker_->setCameraPose(camera_frame);

  // process frame if objects loaded in tracker
  // ------------------------------------------
  if (multi_rigid_tracker_->getNumberOfObjects() > 0) {

    // update detector pose in tracker (if object index valid)
    Ogre::Vector3 camera_position;
    Ogre::Quaternion camera_orientation;
    robot_->getFrame(robot_camera_frame_id_, camera_position,
                     camera_orientation);
    pose::TranslationRotation3D camera_frame(camera_position,
                                             camera_orientation);
    {
      std::lock_guard<std::mutex> lock(most_recent_detector_pose_mutex_);
      if (most_recent_detector_object_index_ <
          multi_rigid_tracker_->getNumberOfObjects()) {
        // transform detector pose to base link
        multi_rigid_tracker_->setRigidDetectorPose(
            camera_frame * most_recent_detector_pose_,
            most_recent_detector_object_index_);
      }
    }

    // update pose
    if (color_only_mode_)
      multi_rigid_tracker_->updatePoses(img_gray);
    else
      multi_rigid_tracker_->updatePoses(img_gray, cv_depth_ptr->image);

    // publish reliable poses on tf
    std::vector<geometry_msgs::Pose> poses =
        multi_rigid_tracker_->getPoseMessages();
    std::vector<std::string> reliably_tracked_objects;

    for (int object_index = 0; object_index < poses.size(); object_index++) {
      if (multi_rigid_tracker_->isPoseReliable(object_index)) {
        geometry_msgs::Pose curr_pose = poses.at(object_index);
        tf::StampedTransform object_transform;
        object_transform.setOrigin(tf::Vector3(
            curr_pose.position.x, curr_pose.position.y, curr_pose.position.z));
        object_transform.setRotation(
            tf::Quaternion(curr_pose.orientation.x, curr_pose.orientation.y,
                           curr_pose.orientation.z, curr_pose.orientation.w));
        object_transform.stamp_ = cv_rgb_ptr->header.stamp;
        object_transform.frame_id_ = "/base_footprint";
        auto object_label = objects_.at(object_index).label_;
        object_transform.child_frame_id_ = object_label;
        reliably_tracked_objects.push_back(object_label);
        tfb_.sendTransform(object_transform);
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

      // store pr2 joint state
      std::vector<std::string> joint_names;
      std::vector<double> joint_angles;
      int n_joints = 0;
      for (auto &it : joint_state_) {
        joint_names.push_back(it.first);
        joint_angles.push_back(it.second);
        n_joints++;
      }
      std::vector<int> joint_size{ n_joints };
      file.writeArray("joint_names", joint_names, joint_size);
      file.writeArray("joint_angles", joint_angles, joint_size);

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

void PR2CamSwitchNode::jointStateCb(const sensor_msgs::JointState &state) {
  if (state.name.size() != state.position.size()) {
    ROS_ERROR("Robot state publisher received an invalid joint state vector");
    return;
  }

  // update joint state
  {
    std::lock_guard<std::mutex> lock(joint_state_mutex_);
    for (int i = 0; i < state.name.size(); i++)
      joint_state_[state.name.at(i)] = state.position.at(i);
  }
}

void
PR2CamSwitchNode::reconfigureCb(simtrack_nodes::VisualizationConfig &config,
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
    std::string robot_description;
    ros::param::get("robot_description", robot_description);
    file.writeScalar("robot_description", robot_description);
    frame_count_ = 0;
    recording_start_time_ = ros::Time::now();
  }

  recording_ = config.start_stop_recording;
}

void PR2CamSwitchNode::setupCameraSubscribers(int camera_index) {

  // unsubscribe from all camera topics
  sync_rgbd_.reset();
  sub_depth_.unsubscribe();
  depth_it_.reset();
  sync_rgb_.reset();
  sub_rgb_info_.unsubscribe();
  sub_rgb_.unsubscribe();
  rgb_it_.reset();

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

  // fetch rgb topic names from parameter server
  std::stringstream topic_name;
  topic_name << "/camera/" << camera_index << "/rgb";
  std::string rgb_topic;
  if (!ros::param::get(topic_name.str(), rgb_topic))
    parameterError(__func__, topic_name.str());
  topic_name.str("");
  topic_name << "/camera/" << camera_index << "/rgb_info";
  std::string rgb_info_topic;
  if (!ros::param::get(topic_name.str(), rgb_info_topic))
    parameterError(__func__, topic_name.str());

  rgb_it_.reset(new image_transport::ImageTransport(nh_));
  sub_rgb_.subscribe(*rgb_it_, rgb_topic, 1, rgb_hint);
  sub_rgb_info_.subscribe(nh_, rgb_info_topic, 1);

  topic_name.str("");
  topic_name << "/camera/" << camera_index << "/robot_frame";
  if (!ros::param::get(topic_name.str(), robot_camera_frame_id_))
    parameterError(__func__, topic_name.str());

  topic_name.str("");
  topic_name << "/camera/" << camera_index << "/color_only_mode";
  if (!ros::param::get(topic_name.str(), color_only_mode_))
    parameterError(__func__, topic_name.str());

  if (color_only_mode_) {
    sync_rgb_.reset(
        new SynchronizerRGB(SyncPolicyRGB(5), sub_rgb_, sub_rgb_info_));
    sync_rgb_->registerCallback(
        boost::bind(&PR2CamSwitchNode::colorOnlyCb, this, _1, _2));
  } else {
    topic_name.str("");
    topic_name << "/camera/" << camera_index << "/depth";
    std::string depth_topic;
    if (!ros::param::get(topic_name.str(), depth_topic))
      parameterError(__func__, topic_name.str());

    depth_it_.reset(new image_transport::ImageTransport(nh_));
    sub_depth_.subscribe(*depth_it_, depth_topic, 1, depth_hint);
    sync_rgbd_.reset(new SynchronizerRGBD(SyncPolicyRGBD(5), sub_depth_,
                                          sub_rgb_, sub_rgb_info_));
    sync_rgbd_->registerCallback(
        boost::bind(&PR2CamSwitchNode::depthAndColorCb, this, _1, _2, _3));
  }
}

void PR2CamSwitchNode::parameterError(std::string function_name,
                                      std::string topic_name) {
  std::stringstream err;
  err << "PR2CamSwitchNode::" << function_name << ": could not find "
      << topic_name << " on parameter server" << std::endl;
  throw std::runtime_error(err.str());
}

interface::MultiRigidTracker::ObjectInfo
PR2CamSwitchNode::composeObjectInfo(std::string model_name) {
  std::string obj_file_name =
      model_path_ + "/" + model_name + "/" + model_name + ".obj";
  return (interface::MultiRigidTracker::ObjectInfo(model_name, obj_file_name));
}

std::string PR2CamSwitchNode::composeObjectFilename(std::string model_name) {
  return (model_path_ + "/" + model_name + "/" + model_name + "_SIFT.h5");
}

} // end namespace simtrack

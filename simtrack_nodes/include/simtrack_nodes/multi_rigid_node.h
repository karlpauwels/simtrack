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

#pragma once

#include <thread>
#include <atomic>
#include <mutex>
#include <boost/filesystem.hpp>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int32.h>
#include <tf/transform_broadcaster.h>
#include <multi_rigid_tracker.h>
#include <multi_rigid_detector.h>
#include <dynamic_reconfigure/server.h>
#include <simtrack_nodes/VisualizationConfig.h>
#include <simtrack_nodes/SwitchObjects.h>

namespace simtrack {

class MultiRigidNode {
public:
  MultiRigidNode(ros::NodeHandle nh);

  ~MultiRigidNode();

  bool start();

private:
  bool switchObjects(simtrack_nodes::SwitchObjectsRequest &req,
                     simtrack_nodes::SwitchObjectsResponse &res);

  void depthAndColorCb(const sensor_msgs::ImageConstPtr &depth_msg,
                       const sensor_msgs::ImageConstPtr &rgb_msg,
                       const sensor_msgs::CameraInfoConstPtr &rgb_info_msg);

  void colorOnlyCb(const sensor_msgs::ImageConstPtr &rgb_msg,
                   const sensor_msgs::CameraInfoConstPtr &rgb_info_msg);

  void reconfigureCb(simtrack_nodes::VisualizationConfig &config,
                     uint32_t level);

  void updatePose(const cv_bridge::CvImagePtr &cv_rgb_ptr,
                  const cv_bridge::CvImagePtr &cv_depth_ptr,
                  const std::string &frame_id);

  void detectorThreadFunction(cv::Mat camera_matrix, size_t width,
                              size_t height);

  interface::MultiRigidTracker::ObjectInfo
  composeObjectInfo(std::string model_name);

  std::string composeObjectFilename(std::string model_name);

  // adjust camera matrix for ROI
  cv::Mat
  composeCameraMatrix(const sensor_msgs::CameraInfoConstPtr &rgb_info_msg);

  std::unique_ptr<std::thread> detector_thread_;
  std::atomic<bool> shutdown_detector_;

  interface::MultiRigidTracker::Ptr multi_rigid_tracker_;
  std::string model_path_;
  std::vector<interface::MultiRigidTracker::ObjectInfo> objects_;
  vision::D_OpticalAndARFlow::Parameters parameters_flow_;
  pose::D_MultipleRigidPoses::Parameters parameters_pose_;

  interface::MultiRigidDetector::Ptr multi_rigid_detector_;
  interface::MultiRigidDetector::Parameters parameters_detector_;
  std::mutex obj_filenames_mutex_;
  std::vector<std::string> obj_filenames_;
  int device_id_detector_;
  std::atomic<bool> detector_enabled_;

  // most recent detector estimate
  std::mutex most_recent_detector_pose_mutex_;
  int most_recent_detector_object_index_;
  pose::TranslationRotation3D most_recent_detector_pose_;

  bool auto_disable_detector_;
  bool color_only_mode_;

  // image used by detector
  std::mutex img_gray_detector_mutex_;
  cv::Mat img_gray_detector_;

  // signals that we just selected objects and will trigger update
  bool switched_tracker_objects_;
  std::atomic<bool> switched_detector_objects_;

  ros::NodeHandle nh_;
  bool ready_;

  // Reconfigure server
  dynamic_reconfigure::Server<simtrack_nodes::VisualizationConfig>
  dynamic_reconfigure_server_;
  interface::MultiRigidTracker::OutputImageType output_image_;
  bool recording_;
  const boost::filesystem::path root_recording_path_;
  boost::filesystem::path recording_path_;
  int frame_count_;
  ros::Time recording_start_time_;
  cv::Mat camera_matrix_rgb_; // for recording purposes only

  struct RecordingFlags {
    RecordingFlags()
        : poses_(false), image_(false), depth_(false), optical_flow_(false) {}
    bool poses_;
    bool image_;
    bool depth_;
    bool optical_flow_;
  };

  RecordingFlags recording_flags_;

  // Services
  ros::ServiceServer switch_objects_srv_;

  // Publishers
  boost::shared_ptr<image_transport::ImageTransport> debug_img_it_;
  image_transport::Publisher debug_img_pub_;
  tf::TransformBroadcaster tfb_;

  // Subscriptions
  ros::Subscriber sub_detector_pose_;
  boost::shared_ptr<image_transport::ImageTransport> rgb_it_, depth_it_;
  image_transport::SubscriberFilter sub_depth_, sub_rgb_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> sub_rgb_info_;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>
  SyncPolicyRGBD;
  typedef message_filters::Synchronizer<SyncPolicyRGBD> SynchronizerRGBD;
  boost::shared_ptr<SynchronizerRGBD> sync_rgbd_;
  typedef message_filters::sync_policies::ApproximateTime<
      sensor_msgs::Image, sensor_msgs::CameraInfo> SyncPolicyRGB;
  typedef message_filters::Synchronizer<SyncPolicyRGB> SynchronizerRGB;
  boost::shared_ptr<SynchronizerRGB> sync_rgb_;
};

} // end namespace simtrack

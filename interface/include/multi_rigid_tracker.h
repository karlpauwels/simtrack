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

#include <bitset>
#include <opencv2/opencv.hpp>
#include <d_optical_and_ar_flow.h>
#include <utility_kernels.h>
#include <d_multiple_rigid_poses.h>
#include <geometry_msgs/Pose.h>
#include <device_1d.h>
#include <device_2d.h>
#include <hdf5_file.h>
#include <d_point_cloud.h>

namespace interface {

class MultiRigidTracker {
public:
  struct ObjectInfo {
    ObjectInfo(std::string label, std::string filename, float scale = 1.0f)
        : label_(label), filename_(filename), scale_(scale) {}
    std::string label_;
    std::string filename_;
    float scale_;
  };

  typedef vision::D_OpticalAndARFlow::Parameters p_flow_t;
  typedef pose::D_MultipleRigidPoses::Parameters p_pose_t;
  MultiRigidTracker(int image_width, int image_height, cv::Mat camera_matrix,
                    std::vector<ObjectInfo> objects,
                    p_flow_t parameters_flow = p_flow_t(),
                    p_pose_t parameters_pose = p_pose_t());

  // remove the rest (rule of five)
  MultiRigidTracker(const MultiRigidTracker &) = delete;
  MultiRigidTracker(MultiRigidTracker &&) = delete;
  MultiRigidTracker &operator=(MultiRigidTracker) = delete;
  MultiRigidTracker &operator=(MultiRigidTracker &&) = delete;

  void setRigidDetectorPose(pose::TranslationRotation3D pose, int object_index);

  void setCameraMatrix(cv::Mat camera_matrix);

  void setCameraPose(const pose::TranslationRotation3D &camera_pose);

  void setWeights(float w_flow, float w_ar_flow, float w_disp);

  void setObjects(std::vector<ObjectInfo> objects);

  int getNumberOfObjects();

  Ogre::SceneManager *getSceneManager() {
    return d_multiple_rigid_poses_->getSceneManager();
  }

  // Estimate pose update from motion only
  void updatePoses(cv::Mat image);

  // Estimate pose update from motion and depth
  void updatePoses(cv::Mat image, cv::Mat depth);

  // Use current depth image from external point cloud to evaluate match with
  // rendered depth
  cv::Mat validatePoses(const pose::D_PointCloud &d_point_cloud,
                        float max_depth_error);

  enum class OutputImageType {
    model_appearance,
    model_appearance_blended,
    optical_flow_x,
    optical_flow_y
  };

  cv::Mat generateOutputImage(OutputImageType image_type);

  std::vector<geometry_msgs::Pose> getPoseMessages();

  std::vector<pose::TranslationRotation3D> getPoses();

  std::vector<std::vector<double> > getBoundingBoxesInCameraFrame() {
    return d_multiple_rigid_poses_->getBoundingBoxesInCameraFrame();
  }

  std::vector<std::vector<double> > getBoundingBoxesInCameraImage() {
    return d_multiple_rigid_poses_->getBoundingBoxesInCameraImage();
  }

  bool isPoseReliable(int object_index);

  bool areAllPosesReliable();

  void saveOpticalFlow(util::HDF5File &file);

  void savePoses(util::HDF5File &file);

  typedef std::unique_ptr<MultiRigidTracker> Ptr;

  const int image_width_;
  const int image_height_;

private:
  void convertImage(cv::Mat image);

  void convertDepth(cv::Mat depth);

  void computePoseUpdate();

  const double baseline_; // set arbitrary value
  double depth_conversion_;

  std::unique_ptr<pose::D_MultipleRigidPoses> d_multiple_rigid_poses_;
  std::unique_ptr<vision::D_OpticalAndARFlow> d_optical_flow_;

  util::Device2D<float>::Ptr d_float_frame_, d_float_ar_frame_;
  util::Device2D<float>::Ptr d_prev_float_frame_, d_disparity_;
  util::Device1D<float>::Ptr d_flow_ar_x_tmp_, d_flow_ar_y_tmp_;
  util::Device1D<uchar4>::Ptr d_flow_x_rgba_, d_flow_y_rgba_;
};
}

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

#include <multi_rigid_tracker.h>
#include <stdexcept>
#undef Success
#include <Eigen/Geometry>
#include <utilities.h>
#include <utility_kernels.h>
#include <utility_kernels_pose.h>
#include <hdf5.h>
#include <numeric>

using namespace util;

namespace interface {

MultiRigidTracker::MultiRigidTracker(int image_width, int image_height,
                                     cv::Mat camera_matrix,
                                     std::vector<ObjectInfo> objects,
                                     p_flow_t parameters_flow,
                                     p_pose_t parameters_pose)
    : image_width_(image_width), image_height_(image_height),
      baseline_(83.0 / 1000.0) {
  d_float_frame_ =
      Device2D<float>::Ptr(new Device2D<float>(image_width_, image_height_));
  d_float_ar_frame_ =
      Device2D<float>::Ptr(new Device2D<float>(image_width_, image_height_));
  d_prev_float_frame_ =
      Device2D<float>::Ptr(new Device2D<float>(image_width_, image_height_));
  d_disparity_ =
      Device2D<float>::Ptr(new Device2D<float>(image_width_, image_height_));
  d_flow_ar_x_tmp_ =
      Device1D<float>::Ptr(new Device1D<float>(image_width_ * image_height_));
  d_flow_ar_y_tmp_ =
      Device1D<float>::Ptr(new Device1D<float>(image_width_ * image_height_));
  d_flow_x_rgba_ =
      Device1D<uchar4>::Ptr(new Device1D<uchar4>(image_width_ * image_height_));
  d_flow_y_rgba_ =
      Device1D<uchar4>::Ptr(new Device1D<uchar4>(image_width_ * image_height_));

  d_optical_flow_ = std::unique_ptr<vision::D_OpticalAndARFlow>{
    new vision::D_OpticalAndARFlow(*d_float_frame_, parameters_flow)
  };

  double nodal_point_x = camera_matrix.at<double>(0, 2);
  double nodal_point_y = camera_matrix.at<double>(1, 2);
  double focal_length_x = camera_matrix.at<double>(0, 0);
  double focal_length_y = camera_matrix.at<double>(1, 1);

  depth_conversion_ = focal_length_x * baseline_;
  double T[]{ 0.1, 0.1, 0.5 };
  double R[]{ 0.78, 0.0, 0.0 };
  pose::TranslationRotation3D init_pose(T, R);

  d_multiple_rigid_poses_ = std::unique_ptr<pose::D_MultipleRigidPoses>(
      new pose::D_MultipleRigidPoses(
          image_width_, image_height_, nodal_point_x, nodal_point_y,
          focal_length_x, focal_length_y, baseline_, parameters_pose));

  for (auto &it : objects)
    d_multiple_rigid_poses_->addModel(it.filename_.c_str(), it.scale_,
                                      init_pose);
}

void MultiRigidTracker::setRigidDetectorPose(pose::TranslationRotation3D pose,
                                             int object_index) {
  d_multiple_rigid_poses_->setSparsePose(pose, object_index);
}

void MultiRigidTracker::setCameraMatrix(cv::Mat camera_matrix) {
  double nodal_point_x = camera_matrix.at<double>(0, 2);
  double nodal_point_y = camera_matrix.at<double>(1, 2);
  double focal_length_x = camera_matrix.at<double>(0, 0);
  double focal_length_y = camera_matrix.at<double>(1, 1);

  d_multiple_rigid_poses_->setCameraParameters(focal_length_x, focal_length_y,
                                               nodal_point_x, nodal_point_y);
}

void MultiRigidTracker::setCameraPose(
    const pose::TranslationRotation3D &camera_pose) {
  d_multiple_rigid_poses_->setCameraPose(camera_pose);
}

void MultiRigidTracker::setObjects(std::vector<ObjectInfo> objects) {
  double T[]{ 0.1, 0.1, 0.5 };
  double R[]{ 0.78, 0.0, 0.0 };
  pose::TranslationRotation3D init_pose(T, R);

  d_multiple_rigid_poses_->removeAllModels();
  for (auto &it : objects)
    d_multiple_rigid_poses_->addModel(it.filename_.c_str(), it.scale_,
                                      init_pose);
}

void MultiRigidTracker::setWeights(float w_flow, float w_ar_flow,
                                   float w_disp) {
  d_multiple_rigid_poses_->setWeights(w_flow, w_ar_flow, w_disp);
}

int MultiRigidTracker::getNumberOfObjects() {
  return (d_multiple_rigid_poses_->getNObjects());
}

void MultiRigidTracker::updatePoses(cv::Mat image) {
  convertImage(image);

  // make sure depth is disabled
  float w_flow, w_ar_flow, w_disp;
  d_multiple_rigid_poses_->getWeights(w_flow, w_ar_flow, w_disp);
  d_multiple_rigid_poses_->setWeights(w_flow, w_ar_flow, 0.0);
  computePoseUpdate();

  // reset weights
  d_multiple_rigid_poses_->setWeights(w_flow, w_ar_flow, w_disp);
}

void MultiRigidTracker::updatePoses(cv::Mat image, cv::Mat depth) {
  convertImage(image);
  convertDepth(depth);
  computePoseUpdate();
}

cv::Mat
MultiRigidTracker::validatePoses(const pose::D_PointCloud &d_point_cloud,
                                 float max_depth_error) {

  // check that the point cloud matches the tracker image size
  if ((d_point_cloud.getNRows() != image_height_) ||
      (d_point_cloud.getNCols() != image_width_)) {
  }

  cv::Mat validation_depth_image = cv::Mat::zeros(
      d_point_cloud.getNRows(), d_point_cloud.getNCols(), CV_32FC1);

  auto &d_depth_image = d_point_cloud.getDeviceDepthImage();

  float llim_depth = 0;
  float ulim_depth = 2;

  auto parameters = d_multiple_rigid_poses_->getParameters();

  pose::colorValidationDepthImageMatches(
      d_flow_x_rgba_->data(), d_depth_image.data(),
      d_multiple_rigid_poses_->getZbuffer(), image_width_, image_height_,
      parameters.near_plane_, parameters.far_plane_, max_depth_error,
      llim_depth, ulim_depth);

  cudaMemcpy(validation_depth_image.data, d_flow_x_rgba_->data(),
             image_width_ * image_height_ * sizeof(uchar4),
             cudaMemcpyDeviceToHost);

  return validation_depth_image;
}

void MultiRigidTracker::convertImage(cv::Mat image) {
  if (image.type() != CV_8UC1)
    throw std::runtime_error(std::string(
        "MultiRigidTracker::convertImage:: image must be CV_8UC1\n"));
  if ((image.rows != image_height_) || (image.cols != image_width_))
    throw std::runtime_error(std::string(
        "MultiRigidTracker::convertImage:: incorrect image size\n"));

  // convert image to float
  cv::Mat image_float;
  image.convertTo(image_float, CV_32FC1);

  // backup previous frame for AR flow image generation
  d_prev_float_frame_->copyFrom(*d_float_frame_);

  // copy new image to device
  cudaMemcpy2D(d_float_frame_->data(), d_float_frame_->pitch(),
               image_float.data, image_width_ * sizeof(float),
               image_width_ * sizeof(float), image_height_,
               cudaMemcpyHostToDevice);
}

void MultiRigidTracker::convertDepth(cv::Mat depth) {
  if ((depth.rows != image_height_) || (depth.cols != image_width_))
    throw std::runtime_error(std::string(
        "MultiRigidTracker::convertDepth:: incorrect depth size\n"));

  // convert depth to float and determine
  // depth -> disparity conversion factor
  cv::Mat depth_32;
  double depth_scale;
  if (depth.type() == CV_32FC1) {
    depth_32 = depth;
    depth_scale = depth_conversion_;
  } else if (depth.type() == CV_16UC1) {
    depth.convertTo(depth_32, CV_32FC1);
    depth_scale = 1000.0 * depth_conversion_;
  } else {
    throw std::runtime_error(
        std::string("MultiRigidTracker::convertDepth:: depth must be CV_32FC1 "
                    "(meter) or CV_16UC1 (millimeter)\n"));
  }

  // copy new depth to device
  cudaMemcpy2D(d_disparity_->data(), d_disparity_->pitch(), depth_32.data,
               image_width_ * sizeof(float), image_width_ * sizeof(float),
               image_height_, cudaMemcpyHostToDevice);

  // convert depth to disparity
  vision::convertKinectDisparityInPlace(d_disparity_->data(),
                                        d_disparity_->pitch(), image_width_,
                                        image_height_, depth_scale);
}

void MultiRigidTracker::computePoseUpdate() {
  //  util::TimerGPU flow_timer;

  // update optical flow
  d_optical_flow_->addImageReal(*d_float_frame_);
  d_optical_flow_->updateOpticalFlowReal();

  //  std::cout << "flow time: " << flow_timer.read() << std::endl;

  // fetch poses
  std::vector<pose::TranslationRotation3D> old_dense_poses =
      d_multiple_rigid_poses_->getPoses();
  std::vector<pose::TranslationRotation3D> old_sparse_poses = old_dense_poses;
  old_sparse_poses.at(d_multiple_rigid_poses_->getSparsePoseObject()) =
      d_multiple_rigid_poses_->getSparsePose();

  // eval sparse
  // -----------

  d_multiple_rigid_poses_->setPoses(old_sparse_poses);

  // compute sparse ar flow
  vision::augmentedRealityFloatArraySelectiveBlend(
      d_float_ar_frame_->data(), d_prev_float_frame_->data(),
      d_multiple_rigid_poses_->getTexture(),
      d_multiple_rigid_poses_->getSegmentIND(), image_width_, image_height_,
      d_float_ar_frame_->pitch(), d_multiple_rigid_poses_->getNObjects());

  d_optical_flow_->addImageAR(*d_float_ar_frame_);
  d_optical_flow_->updateOpticalFlowAR();

  d_multiple_rigid_poses_->evaluateARFlowPoseError(
      false, d_optical_flow_->getARFlowX(), old_sparse_poses);

  // store sparse ar flow to avoid recomputation
  d_flow_ar_x_tmp_->copyFrom(d_optical_flow_->getARFlowX());
  d_flow_ar_y_tmp_->copyFrom(d_optical_flow_->getARFlowY());

  // eval dense
  // -----------

  d_multiple_rigid_poses_->setPoses(old_dense_poses);

  // compute sparse ar flow
  vision::augmentedRealityFloatArraySelectiveBlend(
      d_float_ar_frame_->data(), d_prev_float_frame_->data(),
      d_multiple_rigid_poses_->getTexture(),
      d_multiple_rigid_poses_->getSegmentIND(), image_width_, image_height_,
      d_float_ar_frame_->pitch(), d_multiple_rigid_poses_->getNObjects());

  d_optical_flow_->addImageAR(*d_float_ar_frame_);
  d_optical_flow_->updateOpticalFlowAR();

  d_multiple_rigid_poses_->evaluateARFlowPoseError(
      true, d_optical_flow_->getARFlowX(), old_dense_poses);

  // determine winner
  // ----------------

  if (d_multiple_rigid_poses_->isDenseWinner())
    d_multiple_rigid_poses_->setPoses(old_dense_poses);
  else
    d_multiple_rigid_poses_->setPoses(old_sparse_poses);

  const util::Device1D<float> &d_flow_ar_x_winner =
      d_multiple_rigid_poses_->isDenseWinner() ? d_optical_flow_->getARFlowX()
                                               : *d_flow_ar_x_tmp_.get();
  const util::Device1D<float> &d_flow_ar_y_winner =
      d_multiple_rigid_poses_->isDenseWinner() ? d_optical_flow_->getARFlowY()
                                               : *d_flow_ar_y_tmp_.get();

  //  util::TimerGPU update_timer;
  // update pose
  std::bitset<32> segments_to_update;
  segments_to_update.set();
  d_multiple_rigid_poses_->update(d_optical_flow_->getOpticalFlowX(),
                                  d_optical_flow_->getOpticalFlowY(),
                                  d_flow_ar_x_winner, d_flow_ar_y_winner,
                                  *d_disparity_.get(), segments_to_update);
  //  std::cout << "update time: " << flow_timer.read() << std::endl;
}

cv::Mat MultiRigidTracker::generateOutputImage(OutputImageType image_type) {
  cv::Mat texture = cv::Mat::zeros(image_height_, image_width_, CV_8UC4);

  float lower_lim = -8, upper_lim = 8, min_mag = -1.0f;

  switch (image_type) {
  case OutputImageType::model_appearance:
    // the grayscale texture is rendered between 1 and 2 to easily identify the
    // mask
    d_multiple_rigid_poses_->setRenderStateChanged(true);
    vision::convertFloatArrayToGrayRGBA(d_flow_x_rgba_->data(),
                                        d_multiple_rigid_poses_->getTexture(),
                                        image_width_, image_height_, 1.0, 2.0);
    cudaMemcpy(texture.data, d_flow_x_rgba_->data(),
               image_width_ * image_height_ * sizeof(uchar4),
               cudaMemcpyDeviceToHost);

    break;

  case OutputImageType::model_appearance_blended:
    vision::blendFloatImageFloatArrayToRGBA(
        d_flow_x_rgba_->data(), d_float_frame_->data(),
        d_multiple_rigid_poses_->getTexture(), image_width_ * sizeof(uchar4),
        d_float_frame_->pitch(), image_width_, image_height_);
    cudaMemcpy(texture.data, d_flow_x_rgba_->data(),
               image_width_ * image_height_ * sizeof(uchar4),
               cudaMemcpyDeviceToHost);
    break;

  case OutputImageType::optical_flow_x:
    vision::convertFlowToRGBA(d_flow_x_rgba_->data(), d_flow_y_rgba_->data(),
                              d_optical_flow_->getOpticalFlowX().data(),
                              d_optical_flow_->getOpticalFlowY().data(),
                              image_width_, image_height_, lower_lim, upper_lim,
                              min_mag);
    cudaMemcpy(texture.data, d_flow_x_rgba_->data(),
               image_width_ * image_height_ * sizeof(uchar4),
               cudaMemcpyDeviceToHost);
    break;

  case OutputImageType::optical_flow_y:
    vision::convertFlowToRGBA(d_flow_x_rgba_->data(), d_flow_y_rgba_->data(),
                              d_optical_flow_->getARFlowX().data(),
                              d_optical_flow_->getARFlowY().data(),
                              image_width_, image_height_, lower_lim, upper_lim,
                              min_mag);
    cudaMemcpy(texture.data, d_flow_y_rgba_->data(),
               image_width_ * image_height_ * sizeof(uchar4),
               cudaMemcpyDeviceToHost);
    break;
  }

  return (texture);
}

std::vector<pose::TranslationRotation3D> MultiRigidTracker::getPoses() {
  return d_multiple_rigid_poses_->getPoses();
}

std::vector<geometry_msgs::Pose> MultiRigidTracker::getPoseMessages() {
  std::vector<pose::TranslationRotation3D> my_poses =
      d_multiple_rigid_poses_->getPoses();
  std::vector<geometry_msgs::Pose> ros_poses;
  for (int p = 0; p < my_poses.size(); p++) {
    pose::TranslationRotation3D rightHandedPose = my_poses.at(p);
    double t[3];
    rightHandedPose.getT(t);
    double x, y, z, w;
    rightHandedPose.getQuaternion(x, y, z, w);

    geometry_msgs::Pose pose;
    pose.position.x = t[0];
    pose.position.y = t[1];
    pose.position.z = t[2];
    pose.orientation.x = x;
    pose.orientation.y = y;
    pose.orientation.z = z;
    pose.orientation.w = w;

    ros_poses.push_back(pose);
  }

  return (ros_poses);
}

void MultiRigidTracker::saveOpticalFlow(util::HDF5File &file) {
  std::vector<int> size{ image_height_, image_width_ };
  int n_elements =
      accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
  std::vector<float> flow_x(n_elements);
  std::vector<float> flow_y(n_elements);
  d_optical_flow_->getOpticalFlowX().copyTo(flow_x);
  d_optical_flow_->getOpticalFlowY().copyTo(flow_y);
  file.writeArray("flow_x", flow_x, size);
  file.writeArray("flow_y", flow_y, size);
}

void MultiRigidTracker::savePoses(util::HDF5File &file) {
  std::vector<pose::TranslationRotation3D> my_poses =
      d_multiple_rigid_poses_->getPoses();
  int n_poses = my_poses.size();
  std::vector<int> size{ n_poses, 3 };
  int n_elements =
      accumulate(size.begin(), size.end(), 1, std::multiplies<int>());
  std::vector<double> translations(n_elements);
  std::vector<double> rotations_angle_axis(n_elements);
  std::vector<int> size_mat{ n_poses, 3, 3 };
  int n_elements_mat =
      accumulate(size_mat.begin(), size_mat.end(), 1, std::multiplies<int>());
  std::vector<double> rotations_matrix(n_elements_mat);

  for (int i = 0; i < n_poses; i++) {
    my_poses.at(i).getT(&(translations.at(i * 3)));
    my_poses.at(i).getR(&(rotations_angle_axis.at(i * 3)));
    my_poses.at(i).getR_mat(&(rotations_matrix.at(i * 9)));
  }

  file.writeArray("translations", translations, size);
  file.writeArray("rotations_angle_axis", rotations_angle_axis, size);
  file.writeArray("rotations_matrix", rotations_matrix, size_mat);
}

bool MultiRigidTracker::isPoseReliable(int object_index) {
  return (d_multiple_rigid_poses_->getPoses().at(object_index).isValid());
}

bool MultiRigidTracker::areAllPosesReliable() {
  std::vector<pose::TranslationRotation3D> my_poses =
      d_multiple_rigid_poses_->getPoses();
  int n_poses = my_poses.size();
  int n_valid_poses = 0;
  for (auto &it : my_poses)
    if (it.isValid())
      n_valid_poses++;
  return (n_valid_poses == n_poses);
}
}

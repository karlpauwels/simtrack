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

#include <multi_rigid_detector.h>
#include <stdexcept>
#undef Success
#include <Eigen/Geometry>

namespace interface {

MultiRigidDetector::MultiRigidDetector(int image_width, int image_height,
                                       cv::Mat camera_matrix,
                                       std::vector<std::string> obj_filenames,
                                       int device_id, Parameters parameters)
    : image_width_(image_width), image_height_(image_height),
      parameters_(parameters) {
  double nodal_point_x = camera_matrix.at<double>(0, 2);
  double nodal_point_y = camera_matrix.at<double>(1, 2);
  double focal_length_x = camera_matrix.at<double>(0, 0);
  double focal_length_y = camera_matrix.at<double>(1, 1);

  d_multiple_rigid_pose_sparse_ =
      std::unique_ptr<pose::D_MultipleRigidPoseSparse>(
          new pose::D_MultipleRigidPoseSparse(
              image_width_, image_height_, nodal_point_x, nodal_point_y,
              focal_length_x, focal_length_y, device_id, parameters_.vec_size_,
              parameters_.num_iter_ransac_));

  for (auto &it : obj_filenames)
    d_multiple_rigid_pose_sparse_->addModel(it.c_str());
}

void MultiRigidDetector::estimatePose(const cv::Mat &image, int object_index,
                                      pose::TranslationRotation3D &pose) {
  if (image.type() != CV_8UC1)
    throw std::runtime_error(std::string(
        "MultiRigidDetector::updatePoses:: image must be CV_8UC1\n"));
  if ((image.rows != image_height_) || (image.cols != image_width_))
    throw std::runtime_error(std::string(
        "MultiRigidDetector::updatePoses:: incorrect image size\n"));
  if (object_index >= getNumberOfObjects())
    throw std::runtime_error(std::string(
        "MultiRigidDetector::updatePoses:: object index out of range\n"));

  pose = d_multiple_rigid_pose_sparse_->estimatePoseSpecificObject(
      image, object_index);
}

void MultiRigidDetector::estimatePose(const cv::Mat &image, int object_index,
                                      geometry_msgs::Pose &pose) {
  pose::TranslationRotation3D pose_tr;
  estimatePose(image, object_index, pose_tr);

  double t[3];
  pose_tr.getT(t);
  double x, y, z, w;
  pose_tr.getQuaternion(x, y, z, w);

  pose.position.x = t[0];
  pose.position.y = t[1];
  pose.position.z = t[2];
  pose.orientation.x = x;
  pose.orientation.y = y;
  pose.orientation.z = z;
  pose.orientation.w = w;
}

void MultiRigidDetector::setCameraMatrix(const cv::Mat &camera_matrix) {
  double nodal_point_x = camera_matrix.at<double>(0, 2);
  double nodal_point_y = camera_matrix.at<double>(1, 2);
  double focal_length_x = camera_matrix.at<double>(0, 0);
  double focal_length_y = camera_matrix.at<double>(1, 1);

  d_multiple_rigid_pose_sparse_->updateCalibration(
      image_width_, image_height_, nodal_point_x, nodal_point_y, focal_length_x,
      focal_length_y);
}

void MultiRigidDetector::setObjects(std::vector<std::string> obj_filenames) {
  d_multiple_rigid_pose_sparse_->removeAllModels();
  for (auto &it : obj_filenames)
    d_multiple_rigid_pose_sparse_->addModel(it.c_str());
}

int MultiRigidDetector::getNumberOfObjects() {
  return (d_multiple_rigid_pose_sparse_->getNumberOfObjects());
}
}

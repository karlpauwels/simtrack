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

#include <cstdlib>
#include <cstdio>
#include <iomanip>
#include <chrono>
#include <stdexcept>
#include <Eigen/Dense>
#include <multiple_rigid_models_ogre.h>

namespace pose {

MultipleRigidModelsOgre::MultipleRigidModelsOgre(int image_width,
                                                 int image_height, double fx,
                                                 double fy, double cx,
                                                 double cy, double near_plane,
                                                 double far_plane)
    : image_width_{ image_width }, image_height_{ image_height } {

  ogre_context_ =
      std::unique_ptr<render::OgreContext>{ new render::OgreContext() };

  ogre_multi_render_target_ = std::unique_ptr<render::OgreMultiRenderTarget>{
    new render::OgreMultiRenderTarget("head_mount_kinect", image_width_,
                                      image_height_,
                                      ogre_context_->scene_manager_)
  };

  camera_position_ = Ogre::Vector3(0.0, 0.0, 0.0);
  ;
  camera_orientation_ = Ogre::Quaternion::IDENTITY;
  // convert vision (Z-forward) frame to ogre frame (Z-out)
  camera_orientation_ =
      camera_orientation_ *
      Ogre::Quaternion(Ogre::Degree(180), Ogre::Vector3::UNIT_X);

  updateProjectionMatrix(fx, fy, cx, cy, near_plane, far_plane);

  int n_arrays = 6;
  for (int i = 0; i < n_arrays; i++)
    cuda_gl_interop_arrays_.push_back(new cudaArray *);

  ogre_multi_render_target_->mapCudaArrays(cuda_gl_interop_arrays_);
}

MultipleRigidModelsOgre::~MultipleRigidModelsOgre() {
  ogre_multi_render_target_->unmapCudaArrays();
}

void MultipleRigidModelsOgre::addModel(std::string model_filename) {
  int segment_ind = rigid_objects_.size() + 1;
  std::string model_resource = "file://" + model_filename;
  rigid_objects_.push_back(
      std::unique_ptr<render::RigidObject>{ new render::RigidObject(
          model_resource, ogre_context_->scene_manager_, segment_ind) });
}

void MultipleRigidModelsOgre::removeAllModels() { rigid_objects_.clear(); }

void MultipleRigidModelsOgre::render(
    const std::vector<TranslationRotation3D> &renderPoses) {
  if (renderPoses.size() != rigid_objects_.size())
    throw std::runtime_error("MultipleRigidModelsOgre::render: number of "
                             "renderPoses differs from number of objects\n");

  for (int i = 0; i < renderPoses.size(); i++) {
    bool valid = renderPoses.at(i).isValid();
    rigid_objects_.at(i)->setVisible(valid);
    if (valid) {
      rigid_objects_.at(i)->setPose(renderPoses.at(i).ogreTranslation(),
                                    renderPoses.at(i).ogreRotation());
    }
  }

  ogre_multi_render_target_->unmapCudaArrays();
  ogre_multi_render_target_->render();
  ogre_multi_render_target_->mapCudaArrays(cuda_gl_interop_arrays_);
}

void MultipleRigidModelsOgre::updateProjectionMatrix(double fx, double fy,
                                                     double cx, double cy,
                                                     double near_plane,
                                                     double far_plane) {

  fx_ = fx;
  fy_ = fy;
  cx_ = cx;
  cy_ = cy;
  float zoom_x = 1;
  float zoom_y = 1;

  projection_matrix_ = Ogre::Matrix4::ZERO;
  projection_matrix_[0][0] = 2.0 * fx / (double)image_width_ * zoom_x;
  projection_matrix_[1][1] = 2.0 * fy / (double)image_height_ * zoom_y;
  projection_matrix_[0][2] = 2.0 * (0.5 - cx / (double)image_width_) * zoom_x;
  projection_matrix_[1][2] = 2.0 * (cy / (double)image_height_ - 0.5) * zoom_y;
  projection_matrix_[2][2] =
      -(far_plane + near_plane) / (far_plane - near_plane);
  projection_matrix_[2][3] =
      -2.0 * far_plane * near_plane / (far_plane - near_plane);
  projection_matrix_[3][2] = -1;

  ogre_multi_render_target_->updateCamera(camera_position_, camera_orientation_,
                                          projection_matrix_);
}

void MultipleRigidModelsOgre::updateCamera(Ogre::Vector3 position,
                                           Ogre::Quaternion orientation,
                                           Ogre::Matrix4 projection_matrix) {
  camera_position_ = position;
  camera_orientation_ = orientation;
  projection_matrix_ = projection_matrix;
  ogre_multi_render_target_->updateCamera(camera_position_, camera_orientation_,
                                          projection_matrix_);
}

void MultipleRigidModelsOgre::updateCameraPose(
    const TranslationRotation3D &camera_pose) {
  camera_position_ = camera_pose.ogreTranslation();
  camera_orientation_ = camera_pose.ogreRotation();
  ogre_multi_render_target_->updateCamera(camera_position_, camera_orientation_,
                                          projection_matrix_);
}

std::vector<std::vector<double> >
MultipleRigidModelsOgre::getBoundingBoxesInCameraFrame(
    const std::vector<TranslationRotation3D> &object_poses) {
  if (object_poses.size() != rigid_objects_.size())
    throw std::runtime_error(
        "MultipleRigidModelsOgre::getBoundingBoxesInCameraFrame: "
        "number of renderPoses differs from number of "
        "objects\n");

  auto inv_camera_pose =
      pose::TranslationRotation3D(camera_position_, camera_orientation_)
          .rotateX180()
          .inverseTransform();

  Eigen::Transform<double, 3, Eigen::Affine> t_inv_camera_pose;
  {
    double tra[3];
    double rot[9];
    inv_camera_pose.getT(tra);
    inv_camera_pose.getR_mat(rot);
    Eigen::Translation<double, 3> tra_eigen(tra[0], tra[1], tra[2]);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_eigen(rot);
    t_inv_camera_pose = tra_eigen * rot_eigen;
  }

  std::vector<std::vector<double> > bounding_boxes;

  for (int i = 0; i < object_poses.size(); i++) {
    Eigen::Map<const Eigen::MatrixXf> bb_f(
        rigid_objects_.at(i)->getBoundingBox().data(), 3, 8);

    std::vector<double> bb_vec(3 * 8);
    Eigen::Map<Eigen::Matrix<double, 3, 8> > bounding_box(bb_vec.data());
    bounding_box = bb_f.cast<double>();

    // transform mapped data with object and camera poses
    double tra[3];
    double rot[9];
    object_poses.at(i).getT(tra);
    object_poses.at(i).getR_mat(rot);
    Eigen::Translation<double, 3> tra_eigen(tra[0], tra[1], tra[2]);
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_eigen(rot);
    Eigen::Transform<double, 3, Eigen::Affine> t_object_pose =
        tra_eigen * rot_eigen;
    bounding_box = t_inv_camera_pose * t_object_pose * bounding_box;

    // add to bounding boxes
    bounding_boxes.push_back(bb_vec);
  }

  return bounding_boxes;
}

std::vector<std::vector<double> >
MultipleRigidModelsOgre::getBoundingBoxesInCameraImage(
    const std::vector<TranslationRotation3D> &object_poses) {
  if (object_poses.size() != rigid_objects_.size())
    throw std::runtime_error(
        "MultipleRigidModelsOgre::getBoundingBoxesInCameraImage: "
        "number of renderPoses differs from number of "
        "objects\n");

  auto bounding_boxes_3d = getBoundingBoxesInCameraFrame(object_poses);

  std::vector<std::vector<double> > bounding_boxes;

  for (int i = 0; i < object_poses.size(); i++) {

    Eigen::Map<Eigen::Matrix<double, 3, 8> > bounding_box(
        bounding_boxes_3d.at(i).data());

    // project
    std::vector<double> bb_pixel_vec(2 * 8);
    Eigen::Map<Eigen::Matrix<double, 2, 8> > bb_pixel(bb_pixel_vec.data());
    bb_pixel = bounding_box.block<2, 8>(0, 0);
    // X = Z*(x-ox)/fx -> x = X*fx/Z + ox
    // Y = Z*(y-oy)/fy -> y = Y*fy/Z + oy
    bb_pixel.array() /= bounding_box.row(2).replicate(2, 1).array();
    bb_pixel.row(0) *= fx_;
    bb_pixel.row(1) *= fy_;
    Eigen::Vector2d c;
    c << cx_, cy_;
    bb_pixel.colwise() += c;
    bounding_boxes.push_back(bb_pixel_vec);
  }

  return bounding_boxes;
}

cudaArray *MultipleRigidModelsOgre::getTexture() {
  return (*cuda_gl_interop_arrays_.at(5));
}

cudaArray *MultipleRigidModelsOgre::getZBuffer() {
  return (*cuda_gl_interop_arrays_.at(3));
}

cudaArray *MultipleRigidModelsOgre::getNormalX() {
  return (*cuda_gl_interop_arrays_.at(0));
}

cudaArray *MultipleRigidModelsOgre::getNormalY() {
  return (*cuda_gl_interop_arrays_.at(1));
}

cudaArray *MultipleRigidModelsOgre::getNormalZ() {
  return (*cuda_gl_interop_arrays_.at(2));
}

cudaArray *MultipleRigidModelsOgre::getSegmentIND() {
  return (*cuda_gl_interop_arrays_.at(4));
}
}

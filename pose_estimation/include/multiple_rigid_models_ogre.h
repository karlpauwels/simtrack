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

#include <vector>
#include <memory>

#include <ogre_context.h>
#include <ogre_multi_render_target.h>
#include <rigid_object.h>
#include <translation_rotation_3d.h>

namespace pose {

class MultipleRigidModelsOgre {

public:
  /*! \brief Prepare everything for rendering
  */
  MultipleRigidModelsOgre(int image_width, int image_height, double fx,
                          double fy, double cx, double cy, double near_plane,
                          double far_plane);

  ~MultipleRigidModelsOgre();

  void addModel(std::string model_filename);

  void removeAllModels();

  void render(const std::vector<TranslationRotation3D> &renderPoses);

  void updateProjectionMatrix(double fx, double fy, double cx, double cy,
                              double near_plane, double far_plane);

  void updateCamera(Ogre::Vector3 position, Ogre::Quaternion orientation,
                    Ogre::Matrix4 projection_matrix);

  void updateCameraPose(const TranslationRotation3D &camera_pose);

  /**
  /** * @brief getBoundingBoxesInCameraFrame
  /** * @param object_poses
  /** * @return (8x3) bounding box for each object
  /** */
  std::vector<std::vector<double> > getBoundingBoxesInCameraFrame(
      const std::vector<TranslationRotation3D> &object_poses);

  /**
   * @brief getBoundingBoxesInCameraImage
   * @param object_poses
   * @return (8x2) projected bounding box for each object
   */
  std::vector<std::vector<double> > getBoundingBoxesInCameraImage(
      const std::vector<TranslationRotation3D> &object_poses);

  Ogre::SceneManager *getSceneManager() {
    return ogre_context_->scene_manager_;
  }

  cudaArray *getTexture();
  cudaArray *getZBuffer();
  cudaArray *getNormalX();
  cudaArray *getNormalY();
  /*! \brief The Z component of the normal is expressed in Ogre's coordinate
   * system with -Z forward
  */
  cudaArray *getNormalZ();
  cudaArray *getSegmentIND();

  const render::RigidObject &getRigidObject(int ind) {
    return *rigid_objects_.at(ind);
  }

private:
  const int image_width_;
  const int image_height_;
  Ogre::Matrix4 projection_matrix_;
  double fx_, fy_, cx_, cy_;
  Ogre::Vector3 camera_position_;
  Ogre::Quaternion camera_orientation_;

  std::unique_ptr<render::OgreContext> ogre_context_;
  std::unique_ptr<render::OgreMultiRenderTarget> ogre_multi_render_target_;

  std::vector<std::unique_ptr<render::RigidObject> > rigid_objects_;

  std::vector<cudaArray **> cuda_gl_interop_arrays_;
};
}

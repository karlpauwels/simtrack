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

#include <memory>
#include <translation_rotation_3d.h>
#include <opencv2/core/core.hpp>
#include <SiftGPU.h>

namespace pose {

/* ! \brief Rigid Object Pose Estimation Using Sparse Features
*/
class D_MultipleRigidPoseSparse {

public:
  D_MultipleRigidPoseSparse(int n_cols, int n_rows, float nodal_point_x,
                            float nodal_point_y, float focal_length_x,
                            float focal_length_y, int device_id = 0,
                            int vec_size = 4, int num_iter_ransac = 1000);

  void updateCalibration(int n_cols, int n_rows, float nodal_point_x,
                         float nodal_point_y, float focal_length_x,
                         float focal_length_y);

  void addModel(const char *obj_filename);

  // removes all models from pose estimator
  void removeAllModels();

  // specify for which object the pose should be estimated
  TranslationRotation3D estimatePoseSpecificObject(const cv::Mat &image,
                                                   const int object);

  // randomly select the object (probabilities can be tuned)
  TranslationRotation3D estimatePoseRandomObject(const cv::Mat &image,
                                                 int &object);

  int getNumberOfObjects() { return (_n_objects); }

  void enable() { _running = true; }
  void disable() { _running = false; }

private:
  TranslationRotation3D estimatePose(const cv::Mat &image, int object = 0);

  bool _running;

  cv::Mat _camera_mat;
  int _n_rows, _n_cols;

  const std::unique_ptr<SiftGPU> _siftEngine;
  const std::unique_ptr<SiftMatchGPU> _matcherEngine;
  const int _DESCRIPTOR_LENGTH;
  const int _num_iter_ransac;

  struct ModelAssets {
    int model_size;
    std::vector<float> descriptors;
    std::vector<SiftGPU::SiftKeypoint> positions;
  };
  std::vector<ModelAssets> _allModels;

  int _n_objects;

  std::vector<int> _match_buffer;
  const int _max_matches;
};
}

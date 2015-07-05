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

#include <opencv2/opencv.hpp>
#include <geometry_msgs/Pose.h>
#include <d_multiple_rigid_pose_sparse.h>

namespace interface {

class MultiRigidDetector {
public:
  struct Parameters {
    Parameters() : vec_size_(4), num_iter_ransac_(1000) {}

    int vec_size_;
    int num_iter_ransac_;
  };

  MultiRigidDetector(int image_width, int image_height, cv::Mat camera_matrix,
                     std::vector<std::string> obj_filenames, int device_id,
                     Parameters parameters = Parameters());

  // remove the rest (rule of five)
  MultiRigidDetector(const MultiRigidDetector &) = delete;
  MultiRigidDetector(MultiRigidDetector &&) = delete;
  MultiRigidDetector &operator=(MultiRigidDetector) = delete;
  MultiRigidDetector &operator=(MultiRigidDetector &&) = delete;

  void estimatePose(const cv::Mat &image, int object_index,
                    geometry_msgs::Pose &pose);

  void estimatePose(const cv::Mat &image, int object_index,
                    pose::TranslationRotation3D &pose);

  void setCameraMatrix(const cv::Mat &camera_matrix);

  // removes all objects from detector (if any) and loads new objects
  void setObjects(std::vector<std::string> obj_filenames);

  int getNumberOfObjects();

  typedef std::unique_ptr<MultiRigidDetector> Ptr;

  const int image_width_;
  const int image_height_;

private:
  Parameters parameters_;

  std::unique_ptr<pose::D_MultipleRigidPoseSparse>
  d_multiple_rigid_pose_sparse_;
};
}

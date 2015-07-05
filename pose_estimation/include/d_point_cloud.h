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

#include <cuda_runtime.h>
#include <device_1d.h>
#include <translation_rotation_3d.h>

namespace pose {

class D_PointCloud {

public:
  typedef std::unique_ptr<D_PointCloud> Ptr;

  /**
   * @brief D_PointCloud
   * @param h_point_cloud: interleaved point cloud data with fields
   * x,y,z,rgb(ignored)
   * @param frame_id: name of point cloud coordinate frame
   */
  D_PointCloud(const std::vector<float4> &h_point_cloud, std::string frame_id);

  void updateDepthImage(int n_cols, int n_rows, float nodal_point_x,
                        float nodal_point_y, float focal_length_x,
                        float focal_length_y,
                        pose::TranslationRotation3D transform);

  std::vector<unsigned int> getDepthImage() const;

  int getNRows() const { return n_rows_; }
  int getNCols() const { return n_cols_; }

  const util::Device1D<float> &getDeviceDepthImage() const {
    return *d_depth_image_float_.get();
  }

  /**
   * @brief getDepthImage
   * param h_data: allocated and assumed of size n_rows_*n_cols_
   */
  void getDepthImage(float *h_data) const;

  const std::string frame_id_;

private:
  const util::Device1D<float4>::Ptr d_point_cloud_;

  // depth images, these buffers are dynamically resized if required
  int n_cols_;
  int n_rows_;
  // in millimeter (required for efficient atomic operations)
  util::Device1D<unsigned int>::Ptr d_depth_image_;
  // in meter
  util::Device1D<float>::Ptr d_depth_image_float_;

  // point cloud to camera transform
  const util::Device1D<float>::Ptr d_translation_vector_;
  const util::Device1D<float>::Ptr d_rotation_matrix_;
};

} // end namespace vision

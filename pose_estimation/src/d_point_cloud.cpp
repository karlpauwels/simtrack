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

#include <d_point_cloud.h>
#include <utility_kernels_pose.h>
#include <utilities.h>

namespace pose {

D_PointCloud::D_PointCloud(const std::vector<float4> &h_point_cloud,
                           std::string frame_id)
    : n_cols_{ -1 }, n_rows_{ -1 }, frame_id_{ frame_id },
      d_point_cloud_{ util::Device1D<float4>::make_unique(
          h_point_cloud.size()) },
      d_translation_vector_{ util::Device1D<float>::make_unique(3) },
      d_rotation_matrix_{ util::Device1D<float>::make_unique(3 * 3) } {
  d_point_cloud_->copyFrom(h_point_cloud);
}

void D_PointCloud::updateDepthImage(int n_cols, int n_rows, float nodal_point_x,
                                    float nodal_point_y, float focal_length_x,
                                    float focal_length_y,
                                    pose::TranslationRotation3D transform) {
//  util::TimerGPU timer;

  // reallocate gpu buffer if not suitable
  if ((d_depth_image_.get() == nullptr) || (n_rows != n_rows_) ||
      (n_cols != n_cols_)) {
    n_cols_ = n_cols;
    n_rows_ = n_rows;
    d_depth_image_ =
        util::Device1D<unsigned int>::make_unique(n_cols_ * n_rows_);
    d_depth_image_float_ =
        util::Device1D<float>::make_unique(n_cols_ * n_rows_);
  }

  // copy transform to gpu
  std::vector<float> h_translation_vector(3);
  std::vector<float> h_rotation_matrix(3 * 3);
  transform.getT(h_translation_vector.data());
  transform.getR_mat(h_rotation_matrix.data());
  d_translation_vector_->copyFrom(h_translation_vector);
  d_rotation_matrix_->copyFrom(h_rotation_matrix);

  convertPointCloudToDepthImage(d_depth_image_->data(), d_point_cloud_->data(),
                                n_cols_, n_rows_, d_point_cloud_->size_,
                                nodal_point_x, nodal_point_y, focal_length_x,
                                focal_length_y, d_translation_vector_->data(),
                                d_rotation_matrix_->data());

  convertDepthImageToMeter(d_depth_image_float_->data(), d_depth_image_->data(),
                           n_cols_, n_rows_);

//  std::cout << "point cloud projection time: " << timer.read() << " ms\n";
}

std::vector<unsigned int> D_PointCloud::getDepthImage() const {
  std::vector<unsigned int> h_depth_image(n_cols_ * n_rows_);
  d_depth_image_->copyTo(h_depth_image);
  return h_depth_image;
}

void D_PointCloud::getDepthImage(float *h_data) const {
  cudaMemcpy(h_data, d_depth_image_float_->data(),
             n_rows_ * n_cols_ * sizeof(float), cudaMemcpyDeviceToHost);
}

} // end namespace vision

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

#include <device_1d.h>
#include <device_2d.h>
#include <d_gabor_pyramid.h>

namespace vision {

/*! \brief Flow and stereo optimized for pose estimation (two frame flow, stereo
 * priors, multiple ar images, ...)
 */
class D_OpticalAndARFlow {

public:
  struct Parameters {
    Parameters()
        : n_scales_(0), median_filter_(true), consistent_(true),
          cons_thres_(0.5f), four_orientations_(false) {}
    int n_scales_; // default to max scales
    bool median_filter_;
    bool consistent_;
    float cons_thres_;
    bool four_orientations_;
  };

  D_OpticalAndARFlow(const util::Device2D<float> &rgb,
                     Parameters parameters = Parameters());

  ~D_OpticalAndARFlow();

  void addImageReal(const util::Device2D<float> &rgb);
  void addImageAR(const util::Device2D<float> &rgb);

  void updateOpticalFlowReal();
  void updateOpticalFlowAR();

  const util::Device1D<float> &getOpticalFlowX() const {
    return *optical_flow_real_x_.get();
  }
  const util::Device1D<float> &getOpticalFlowY() const {
    return *optical_flow_real_y_.get();
  }
  const util::Device1D<float> &getARFlowX() const {
    return *optical_flow_ar_x_.get();
  }
  const util::Device1D<float> &getARFlowY() const {
    return *optical_flow_ar_y_.get();
  }

  const int width_;
  const int height_;

private:
  // resize image to root image by border replication
  void updateRootImage(const util::Device2D<float> &image);

  Parameters parameters_;

  // internal size can differ in order to accommodate n_scales
  int width_internal_;
  int height_internal_;

  // root image, possibly resized to accomodate n_scales
  std::shared_ptr<util::Device2D<float> > root_image_;

  // buffer shared by all gaborpyramids
  std::shared_ptr<util::Device2D<char> > buffer_;
  // each image reuses the pyramid
  std::unique_ptr<D_ImagePyramid> image_pyramid_;
  // real images gabor pyramids (previous and most recent frame)
  std::vector<std::unique_ptr<D_GaborPyramid> > gabor_pyramids_real_;
  // ar image gabor pyramid (re-used)
  std::unique_ptr<D_GaborPyramid> gabor_pyramid_ar_;

  // multiscale optical flow estimates (internal resolution)
  std::vector<util::Device2D<float2>::Ptr> optical_flow_pyramid_;

  // de-interleaved flow at image resolution
  util::Device1D<float>::Ptr optical_flow_real_x_;
  util::Device1D<float>::Ptr optical_flow_real_y_;
  util::Device1D<float>::Ptr optical_flow_ar_x_;
  util::Device1D<float>::Ptr optical_flow_ar_y_;

  // used for consistency check
  cudaArray *_d_frame2FlowArray;
};

} // end namespace vision

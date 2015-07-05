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

#include <vector>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <d_optical_and_ar_flow.h>
#include <optical_flow_kernels.h>
#include <convolution_kernels.h>
#include <utility_kernels.h>

namespace vision {
D_OpticalAndARFlow::D_OpticalAndARFlow(const util::Device2D<float> &rgb,
                                       Parameters parameters)
    : width_{ rgb.width_ }, height_{ rgb.height_ }, parameters_{ parameters },
      width_internal_{ rgb.width_ }, height_internal_{ rgb.height_ } {
  int max_scales =
      vision::D_GaborPyramid::computeMaxNumberOfScales(width_, height_);
  if (parameters_.n_scales_ > max_scales) {
    std::stringstream error_string;
    error_string << "D_OpticalFlowAndKinectDepthForPoseEstimation::D_"
                    "OpticalFlowAndKinectDepthForPoseEstimation; number of "
                    "scales too large to accommodate filter kernel at lowest "
                    "resolution (max scales = " << max_scales << ")";
    throw std::runtime_error(error_string.str());
  }

  // use max scales by default
  if (parameters_.n_scales_ == 0)
    parameters_.n_scales_ = max_scales;

  // determine width and height that can accomodate n_scales and will be used
  // internally
  vision::D_GaborPyramid::computeCompatibleImageSize(
      width_internal_, height_internal_, parameters_.n_scales_);

  // construct root image of appropriate size
  root_image_ = std::make_shared<util::Device2D<float> >(width_internal_,
                                                         height_internal_);
  updateRootImage(rgb);

  image_pyramid_ = std::unique_ptr<D_ImagePyramid>{ new D_ImagePyramid(
      root_image_, parameters_.n_scales_) };

  // allocate largest block of temporary memory needed by D_GaborPyramid
  buffer_ = D_GaborPyramid::makeTempBuffer(width_internal_, height_internal_,
                                           parameters_.four_orientations_);

  gabor_pyramids_real_.push_back(std::unique_ptr<D_GaborPyramid>{
    new D_GaborPyramid(*image_pyramid_, buffer_, parameters_.four_orientations_)
  });
  gabor_pyramids_real_.push_back(std::unique_ptr<D_GaborPyramid>{
    new D_GaborPyramid(*image_pyramid_, buffer_, parameters_.four_orientations_)
  });
  gabor_pyramid_ar_ = std::unique_ptr<D_GaborPyramid>{ new D_GaborPyramid(
      *image_pyramid_, buffer_, parameters_.four_orientations_) };

  // allocate device optical flow pyramid (re-used for real and ar)
  for (int s = 0; s < image_pyramid_->n_scales_; s++) {
    auto &image = image_pyramid_->getImageAtScale(s);
    optical_flow_pyramid_.push_back(
        util::Device2D<float2>::make_unique(image.width_, image.height_));
  }

  // allocate space for de-interleaved real and ar flow
  optical_flow_real_x_ = util::Device1D<float>::make_unique(width_ * height_);
  optical_flow_real_y_ = util::Device1D<float>::make_unique(width_ * height_);
  optical_flow_ar_x_ = util::Device1D<float>::make_unique(width_ * height_);
  optical_flow_ar_y_ = util::Device1D<float>::make_unique(width_ * height_);

  // allocate space on device for flow array (consistency check)
  cudaChannelFormatDesc channelFloat2 = cudaCreateChannelDesc<float2>();
  cudaMallocArray(&_d_frame2FlowArray, &channelFloat2, width_internal_,
                  height_internal_);
}

D_OpticalAndARFlow::~D_OpticalAndARFlow() { cudaFreeArray(_d_frame2FlowArray); }

void D_OpticalAndARFlow::addImageReal(const util::Device2D<float> &rgb) {
  if ((width_ != rgb.width_) || (height_ != rgb.height_))
    throw std::runtime_error("D_OpticalFlowAndKinectDepthForPoseEstimation::"
                             "addImageReal; rgb size cannot change.\n");

  updateRootImage(rgb);

  // update circular buffer
  gabor_pyramids_real_.at(0).swap(gabor_pyramids_real_.at(1));

  // update frame 1 gaborpyramid
  image_pyramid_->resetRootImage(root_image_);
  gabor_pyramids_real_.at(1)->resetImagePyramid(*image_pyramid_);
}

void D_OpticalAndARFlow::addImageAR(const util::Device2D<float> &rgb) {
  if ((width_ != rgb.width_) || (height_ != rgb.height_))
    throw std::runtime_error("D_OpticalFlowAndKinectDepthForPoseEstimation::"
                             "addImageAR; rgb size cannot change.\n");

  updateRootImage(rgb);

  image_pyramid_->resetRootImage(root_image_);
  gabor_pyramid_ar_->resetImagePyramid(*image_pyramid_);
}

void D_OpticalAndARFlow::updateOpticalFlowReal() {

  std::vector<PitchFloat2Mem> gabPyrReal0(parameters_.n_scales_);
  std::vector<PitchFloat2Mem> gabPyrReal1(parameters_.n_scales_);
  std::vector<PitchFloat2Mem> optFlowPyr(parameters_.n_scales_);
  std::vector<int> n_rows(parameters_.n_scales_);
  std::vector<int> n_cols(parameters_.n_scales_);

  for (int s = 0; s < parameters_.n_scales_; s++) {
    auto& gab0 =
        gabor_pyramids_real_.at(0)->getGaborAtScale(s).getGaborInterleaved();
    auto& gab1 =
        gabor_pyramids_real_.at(1)->getGaborAtScale(s).getGaborInterleaved();
    gabPyrReal0.at(s).ptr = gab0.data();
    gabPyrReal0.at(s).pitch = gab0.pitch();
    gabPyrReal1.at(s).ptr = gab1.data();
    gabPyrReal1.at(s).pitch = gab1.pitch();
    optFlowPyr.at(s).ptr = optical_flow_pyramid_.at(s)->data();
    optFlowPyr.at(s).pitch = optical_flow_pyramid_.at(s)->pitch();
    n_rows.at(s) = image_pyramid_->getImageAtScale(s).height_;
    n_cols.at(s) = image_pyramid_->getImageAtScale(s).width_;
  }

  computeOpticalFlowTwoFrames(optFlowPyr, buffer_->data(), buffer_->pitch(),
                              gabPyrReal0, gabPyrReal1, _d_frame2FlowArray,
                              parameters_.n_scales_, parameters_.median_filter_,
                              parameters_.consistent_, parameters_.cons_thres_,
                              n_rows, n_cols, parameters_.four_orientations_);

  deInterleave(optical_flow_real_x_->data(), optical_flow_real_y_->data(),
               optical_flow_pyramid_.at(parameters_.n_scales_ - 1)->data(),
               width_ * sizeof(float),
               optical_flow_pyramid_.at(parameters_.n_scales_ - 1)->pitch(),
               width_, height_);
}

void D_OpticalAndARFlow::updateOpticalFlowAR() {

  std::vector<PitchFloat2Mem> gabPyrReal0(parameters_.n_scales_);
  std::vector<PitchFloat2Mem> gabPyrReal1(parameters_.n_scales_);
  std::vector<PitchFloat2Mem> optFlowPyr(parameters_.n_scales_);
  std::vector<int> n_rows(parameters_.n_scales_);
  std::vector<int> n_cols(parameters_.n_scales_);

  for (int s = 0; s < parameters_.n_scales_; s++) {
    auto& gab0 = gabor_pyramid_ar_->getGaborAtScale(s).getGaborInterleaved();
    auto& gab1 =
        gabor_pyramids_real_.at(1)->getGaborAtScale(s).getGaborInterleaved();
    gabPyrReal0.at(s).ptr = gab0.data();
    gabPyrReal0.at(s).pitch = gab0.pitch();
    gabPyrReal1.at(s).ptr = gab1.data();
    gabPyrReal1.at(s).pitch = gab1.pitch();
    optFlowPyr.at(s).ptr = optical_flow_pyramid_.at(s)->data();
    optFlowPyr.at(s).pitch = optical_flow_pyramid_.at(s)->pitch();
    n_rows.at(s) = image_pyramid_->getImageAtScale(s).height_;
    n_cols.at(s) = image_pyramid_->getImageAtScale(s).width_;
  }

  computeOpticalFlowTwoFrames(optFlowPyr, buffer_->data(), buffer_->pitch(),
                              gabPyrReal0, gabPyrReal1, _d_frame2FlowArray,
                              parameters_.n_scales_, parameters_.median_filter_,
                              parameters_.consistent_, parameters_.cons_thres_,
                              n_rows, n_cols, parameters_.four_orientations_);

  deInterleave(optical_flow_ar_x_->data(), optical_flow_ar_y_->data(),
               optical_flow_pyramid_.at(parameters_.n_scales_ - 1)->data(),
               width_ * sizeof(float),
               optical_flow_pyramid_.at(parameters_.n_scales_ - 1)->pitch(),
               width_, height_);
}

void D_OpticalAndARFlow::updateRootImage(const util::Device2D<float> &image) {
  vision::resize_replicate_border(
      image.data(), image.pitch(), root_image_->data(), root_image_->pitch(),
      image.width_, image.height_, root_image_->width_, root_image_->height_);
}

} // end namespace vision

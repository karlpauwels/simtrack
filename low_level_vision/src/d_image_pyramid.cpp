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

#include <cstdio>
#include <stdexcept>
#include <d_image_pyramid.h>
#include <convolution_kernels.h>

namespace vision {

D_ImagePyramid::D_ImagePyramid(std::shared_ptr<util::Device2D<float> > image,
                               int n_scales)
    : root_image_{ image }, n_scales_{ n_scales } {
  downsampled_images_.resize(n_scales_ - 1);

  int w = root_image_->width_;
  int h = root_image_->height_;

  for (int s = (n_scales_ - 2); s >= 0; s--) {
    // check that the number of scales requested is compatible with the image
    // size
    // we need to be able to half the image size (n_scales-1) times
    if (((w % 2) != 0) || ((h % 2) != 0))
      throw std::runtime_error("D_ImagePyramid::D_ImagePyramid; image size "
                               "incompatible with number of scales! Both width "
                               "and height need to have 2^(n_scales-1) as "
                               "factor.\n");
    w /= 2;
    h /= 2;
    downsampled_images_.at(s) = util::Device2D<float>::make_unique(w, h);
  }

  downSampleImages();
}

void
D_ImagePyramid::resetRootImage(std::shared_ptr<util::Device2D<float> > image) {
  // check size compatibility
  if ((root_image_->width_ != image->width_) ||
      (root_image_->height_ != image->height_))
    throw std::runtime_error(
        "D_ImagePyramid::resetRootImage; image size cannot change on reset.\n");
  root_image_ = image;

  downSampleImages();
}

void D_ImagePyramid::downSampleImages() {
  // root image is a special case (shared instead of unique pointer)
  if (n_scales_ > 1) {
    auto &i0 = root_image_;
    auto &i1 = downsampled_images_.at(n_scales_ - 2);
    downSample(i0->data(), i0->pitch(), i1->data(), i1->pitch(), i0->width_,
               i0->height_);
  }

  for (int s = (n_scales_ - 2); s > 0; s--) // not at lowest scale
  {
    auto &i0 = downsampled_images_.at(s);
    auto &i1 = downsampled_images_.at(s - 1);
    downSample(i0->data(), i0->pitch(), i1->data(), i1->pitch(), i0->width_,
               i0->height_);
  }
}

const util::Device2D<float> &D_ImagePyramid::getImageAtScale(int scale) const {
  if ((scale < 0) || (scale >= n_scales_))
    throw std::runtime_error("D_ImagePyramid::getImage; invalid scale\n");

  if (scale == (n_scales_ - 1))
    return *root_image_.get();
  else
    return *downsampled_images_.at(scale).get();
}

} // end namespace vision

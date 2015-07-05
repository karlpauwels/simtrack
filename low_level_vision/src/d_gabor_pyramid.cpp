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

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <d_gabor_pyramid.h>

namespace vision {

int D_GaborPyramid::computeMaxNumberOfScales(int image_width,
                                             int image_height) {
  int n_scales_width = floor(log2(image_width / (double)filter_size_)) + 1;
  int n_scales_height = floor(log2(image_height / (double)filter_size_)) + 1;
  return std::min<int>(n_scales_width, n_scales_height);
}

void D_GaborPyramid::computeCompatibleImageSize(int &image_width,
                                                int &image_height,
                                                int n_scales) {
  double factor = pow(2.0, double(n_scales - 1));
  image_width = ceil(image_width / factor) * factor;
  image_height = ceil(image_height / factor) * factor;
}

std::shared_ptr<util::Device2D<char> >
D_GaborPyramid::makeTempBuffer(int image_width, int image_height,
                               bool four_orientations) {
  int buffer_width = image_width * sizeof(float2);
  int buffer_height = four_orientations ? image_height * 3 : image_height * 5;
  return (std::make_shared<util::Device2D<char> >(buffer_width, buffer_height));
}

D_GaborPyramid::D_GaborPyramid(const D_ImagePyramid &image_pyramid,
                               std::shared_ptr<util::Device2D<char> > buffer,
                               bool four_orientations)
    : buffer_{ buffer } {
  // allocate space for the responses and filter the images
  gabors_.resize(image_pyramid.n_scales_);
  for (int s = 0; s < image_pyramid.n_scales_; s++)
    gabors_.at(s) = D_Gabor::Ptr{ new D_Gabor(image_pyramid.getImageAtScale(s),
                                              buffer_, four_orientations) };
}

void D_GaborPyramid::resetImagePyramid(const D_ImagePyramid &image_pyramid) {
  if (image_pyramid.n_scales_ != gabors_.size())
    throw std::runtime_error(
        "D_GaborPyramid::resetImagePyramid; invalid number of scales.\n");

  for (int s = 0; s < image_pyramid.n_scales_; s++)
    gabors_.at(s)->resetImage(image_pyramid.getImageAtScale(s));
}

const D_Gabor &D_GaborPyramid::getGaborAtScale(int scale) const {
  if ((scale < 0) || (scale >= gabors_.size()))
    throw std::runtime_error(
        "D_GaborPyramid::getGaborAtScale; invalid scale\n");

  return *gabors_.at(scale).get();
}

} // end namespace vision

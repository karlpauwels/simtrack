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

#include <d_gabor.h>
#include <convolution_kernels.h>

namespace vision {

D_Gabor::D_Gabor(const util::Device2D<float> &image,
                 std::shared_ptr<util::Device2D<char> > buffer,
                 bool four_orientations)
    : buffer_{ buffer }, four_orientations_{ four_orientations },
      width_{ image.width_ }, height_{ image.height_ },
      gabor_interleaved_{ util::Device2D<float2>::make_unique(
          width_, height_ * (four_orientations_ ? 4 : 8)) } {
  // check buffer size
  if (four_orientations_) {
    if ((buffer_->width_ < width_ * sizeof(float2)) ||
        (buffer_->height_ < 3 * height_))
      throw std::runtime_error("D_Gabor::D_Gabor; buffer size insufficient.\n");
  } else {
    if ((buffer_->width_ < width_ * sizeof(float2)) ||
        (buffer_->height_ < 5 * height_))
      throw std::runtime_error("D_Gabor::D_Gabor; buffer size insufficient.\n");
  }

  resetImage(image);
}

void D_Gabor::resetImage(const util::Device2D<float> &image) {
  if ((width_ != image.width_) || (height_ != image.height_))
    throw std::runtime_error(
        "D_Gabor::resetImage; image size cannot change on reset.\n");

  gaborFilterItl(image.data(), image.pitch(), gabor_interleaved_->data(),
                 gabor_interleaved_->pitch(), buffer_->data(), buffer_->pitch(),
                 width_, height_, four_orientations_);
}

} // end namespace vision

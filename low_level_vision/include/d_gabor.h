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
#include <device_2d.h>

namespace vision {

/*! \brief Device Gabor Class

    Performs Gabor filtering on GPU
 */
class D_Gabor {

public:
  /*! \brief Construct D_Gabor object and filter image at eight orientations

      \param image The device image object
      \param d_TEMP Temporary space used for filtering. Should be of size [
     width*sizeof(float2) ] x [ height*5 ] in case of eight orientations and of
     size [ width*sizeof(float2) ] x [ height*3 ] in case of four orientations
      \param fourOrientations If true, filtering will be performed at four
     orientation only
   */
  D_Gabor(const util::Device2D<float> &image,
          std::shared_ptr<util::Device2D<char> > buffer,
          bool four_orientations = false);

  typedef std::unique_ptr<D_Gabor> Ptr;

  /*! \brief Gabor filter image. Allows re-use of D_Gabor object */
  void resetImage(const util::Device2D<float> &image);

  /*! \brief Get access to device Gabor data */
  const util::Device2D<float2> &getGaborInterleaved() const {
    return *gabor_interleaved_.get();
  }

private:
  const bool four_orientations_;
  const int width_;
  const int height_;
  const util::Device2D<float2>::Ptr gabor_interleaved_;
  const std::shared_ptr<util::Device2D<char> > buffer_;
};

} // end namespace vision

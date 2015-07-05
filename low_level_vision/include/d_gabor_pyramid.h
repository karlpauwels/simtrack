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

#include <device_2d.h>
#include <d_gabor.h>
#include <d_image_pyramid.h>

namespace vision {

/*! \brief Device Gabor Pyramid Class

    Computes and stores entire Gabor Pyramid on the basis of an Image Pyramid
*/
class D_GaborPyramid {

public:
  /*!
   * \brief maxNumberOfScales Computes maximal number of scales that can
   * accommodate the 11x11 filters
   * \param image_width
   * \param image_height
   * \return
   */
  static int computeMaxNumberOfScales(int image_width, int image_height);

  /*!
   * \brief computeCompatibleImageSize Computes closest (larger) image size that
   * has 2^(n_scales-1) as factor
   * \param image_width
   * \param image_height
   * \param n_scales
   */
  static void computeCompatibleImageSize(int &image_width, int &image_height,
                                         int n_scales);

  /*!
   * \brief allocateTempBuffer Allocates external device memory buffer of
   * required size
   *  that can be re-used by multiple D_GaborPyramids (not thread-safe)
   *  Note: There is no static shared_ptr associated with this buffer */
  static std::shared_ptr<util::Device2D<char> >
  makeTempBuffer(int image_width, int image_height, bool four_orientations);

  /*! \brief Construct a new Gabor pyramid

      Allocates space for all the responses and filters the images contained in
     the image pyramid

      \param d_TEMP Temporary space used for filtering. Should be of size [
     width*sizeof(float2) ] x [ height*5 ] in case of eight orientations and of
     size [ width*sizeof(float2) ] x [ height*3 ] in case of four orientations
      \param fourOrientations If true, filtering will be performed at four
     orientation only
   */
  D_GaborPyramid(const D_ImagePyramid &image_pyramid,
                 std::shared_ptr<util::Device2D<char> > buffer,
                 bool four_orientations = false);

  /*! \brief Gabor filter new image pyramid without re-allocating memory

      Allows for re-use of existing D_GaborPyramid structure, assuming the new
     image pyramid has the same dimensions and number of scales
   */
  void resetImagePyramid(const D_ImagePyramid &image_pyramid);

  /*! Return D_Gabor object at requested scale */
  const D_Gabor &getGaborAtScale(int scale) const;

private:
  static const int filter_size_ = 11;
  std::vector<D_Gabor::Ptr> gabors_;
  const std::shared_ptr<util::Device2D<char> > buffer_;
};

} // end namespace vision

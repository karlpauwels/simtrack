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

#include <memory>
#include <stdexcept>
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <device_2d.h>
#include <d_image_pyramid.h>
#include <d_gabor_pyramid.h>
#include <d_optical_and_ar_flow.h>

TEST(TestLowLevel, initializeCUDA) {
  cudaError_t err = cudaSetDevice(0);
  ASSERT_EQ(cudaSuccess, err) << cudaGetErrorString(err);
}

TEST(TestD_ImagePyramid, sharedPointerStorage) {
  int width = 640;
  int height = 480;

  auto image = std::make_shared<util::Device2D<float> >(width, height);

  int n_scales = 6;
  vision::D_ImagePyramid pyramid(image, n_scales);

  image.reset();
  auto new_image = std::make_shared<util::Device2D<float> >(width, height);
  //  std::cout << "before reset\n";
  pyramid.resetRootImage(new_image);
  //  std::cout << "after reset\n";

  auto &ret_image = pyramid.getImageAtScale(2);
}

TEST(TestD_GaborPyramid, sharedPointerStorage) {
  int width = 640;
  int height = 480;

  auto image = std::make_shared<util::Device2D<float> >(width, height);

  int n_scales = 6;
  vision::D_ImagePyramid pyramid(image, n_scales);

  bool four_orientations = false;
  auto buffer =
      vision::D_GaborPyramid::makeTempBuffer(width, height, four_orientations);

  vision::D_GaborPyramid gab_pyr(pyramid, buffer, four_orientations);

  gab_pyr.resetImagePyramid(pyramid);
}

TEST(TestOpticalFlow, sharedPointerStorage) {
  int width = 640;
  int height = 480;

  util::Device2D<float> rgb(width, height);

  vision::D_OpticalAndARFlow opt_flow(rgb);
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

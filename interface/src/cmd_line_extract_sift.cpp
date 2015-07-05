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

#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <hdf5_file.h>
#include <SiftGPU.h>
#include <GL/gl.h>

int main(int argc, char **argv) {

  /*********/
  /* INPUT */
  /*********/

  if (argc < 3)
    throw std::runtime_error(
        "Usage: ./cmd_line_extract_sift <input.h5> <output.h5>");

  // create files
  util::HDF5File in_file(argv[1]);
  util::HDF5File out_file(argv[2]);

  // parse inputs
  std::vector<int> image_size;
  std::vector<uint8_t> image_data;

  in_file.readArray("image", image_data, image_size);

  if (image_size.size() != 2)
    throw std::runtime_error("Expecting 2D uint8_t image");

  /***********/
  /* PROCESS */
  /***********/

  SiftGPU sift_engine;

  const char *argv_sift[] = { "-m", "-fo",   "-1",    "-s",    "-v",
                              "0",  "-pack", "-cuda", "-maxd", "3840" };

  int argc_sift = sizeof(argv_sift) / sizeof(char *);
  sift_engine.ParseParam(argc_sift, (char **)argv_sift);

  if (sift_engine.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    throw std::runtime_error("SIFT cannot create GL context");

  bool success =
      sift_engine.RunSIFT(image_size.at(1), image_size.at(0), image_data.data(),
                          GL_LUMINANCE, GL_UNSIGNED_BYTE);

  if (!success)
    throw std::runtime_error("SIFT run failed");

  int num_features = sift_engine.GetFeatureNum();
  std::vector<float> descriptors_data(num_features * 128);
  std::vector<SiftGPU::SiftKeypoint> keys(num_features);
  sift_engine.GetFeatureVector(keys.data(), descriptors_data.data());

  std::vector<float> keys_data(num_features * 4);
  for (int i = 0; i < num_features; i++) {
    keys_data.at(i * 4) = keys.at(i).x;
    keys_data.at(i * 4 + 1) = keys.at(i).y;
    keys_data.at(i * 4 + 2) = keys.at(i).s;
    keys_data.at(i * 4 + 3) = keys.at(i).o;
  }

  /**********/
  /* OUTPUT */
  /**********/

  std::vector<int> descriptors_size{ num_features, 128 };
  out_file.writeArray("descriptors", descriptors_data, descriptors_size);
  std::vector<int> keys_size{ num_features, 4 };
  out_file.writeArray("keys", keys_data, keys_size);

  return EXIT_SUCCESS;
}

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

#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <hdf5_file.h>
#include <device_2d.h>
#include <utilities.h>
#include <d_optical_and_ar_flow.h>

template <class Type>
Type fetchScalar(util::HDF5File &in_file, std::string name,
                 Type default_value) {
  return in_file.checkVariableExists(name) ? in_file.readScalar<Type>(name)
                                           : default_value;
}

int main(int argc, char **argv) {

  /*********/
  /* INPUT */
  /*********/

  if (argc < 3)
    throw std::runtime_error(
        "Usage: ./cmd_line_optical_flow <input.h5> <output.h5>");

  // create files
  util::HDF5File in_file(argv[1]);
  util::HDF5File out_file(argv[2]);

  std::vector<int> image0_size, image1_size;
  std::vector<float> image0_data, image1_data;

  in_file.readArray("image0", image0_data, image0_size);
  in_file.readArray("image1", image1_data, image1_size);

  if (image0_size.size() != 2)
    throw std::runtime_error("Expecting 2D float image");

  if (image0_size != image1_size)
    throw std::runtime_error("Expecting equal size images");

  // default arguments
  vision::D_OpticalAndARFlow::Parameters parameters;

  parameters.n_scales_ =
      fetchScalar<int>(in_file, "n_scales", parameters.n_scales_);
  parameters.median_filter_ = (bool)fetchScalar<int>(in_file, "median_filter",
                                                     parameters.median_filter_);
  parameters.consistent_ =
      (bool)fetchScalar<int>(in_file, "consistent", parameters.consistent_);
  parameters.cons_thres_ =
      fetchScalar<float>(in_file, "cons_thres", parameters.cons_thres_);
  parameters.four_orientations_ = fetchScalar<int>(
      in_file, "four_orientations", parameters.four_orientations_);

  // time execution?
  bool timing = (bool)fetchScalar<int>(in_file, "timing", false);

  /***********/
  /* PROCESS */
  /***********/

  int width = image0_size.at(1);
  int height = image0_size.at(0);

  // this run also serves as warm-up for the timing code
  util::Device2D<float> d_image0(width, height);
  d_image0.copyFrom(image0_data);
  util::Device2D<float> d_image1(width, height);
  d_image1.copyFrom(image1_data);
  vision::D_OpticalAndARFlow optical_flow(d_image0, parameters);
  optical_flow.addImageReal(d_image1);
  optical_flow.updateOpticalFlowReal();

  // output already since timing will overwrite
  auto &flow_x = optical_flow.getOpticalFlowX();
  std::vector<int> h_size{ height, width };
  std::vector<float> h_data(h_size.at(0) * h_size.at(1));
  flow_x.copyTo(h_data);
  out_file.writeArray("optical_flow_x", h_data, h_size);
  auto &flow_y = optical_flow.getOpticalFlowY();
  flow_y.copyTo(h_data);
  out_file.writeArray("optical_flow_y", h_data, h_size);

  // timing
  float image_copy_time, gabor_time, flow_time;
  if (timing) {
    int n_reps = 10;
    util::TimerGPU timer;

    // image copy
    timer.reset();
    for (int r = 0; r < n_reps; r++)
      d_image1.copyFrom(image1_data);
    image_copy_time = timer.read() / (float)n_reps;

    // gabor filtering
    timer.reset();
    for (int r = 0; r < n_reps; r++)
      optical_flow.addImageReal(d_image1);
    gabor_time = timer.read() / (float)n_reps;

    // optical flow
    timer.reset();
    for (int r = 0; r < n_reps; r++)
      optical_flow.updateOpticalFlowReal();
    flow_time = timer.read() / (float)n_reps;

    // output timers
    out_file.writeScalar("image_copy_time", image_copy_time);
    out_file.writeScalar("gabor_time", gabor_time);
    out_file.writeScalar("flow_time", flow_time);
  }

  return EXIT_SUCCESS;
}

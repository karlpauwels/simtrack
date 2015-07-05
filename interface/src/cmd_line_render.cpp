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
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <hdf5_file.h>
#include <device_1d.h>
#include <multiple_rigid_models_ogre.h>
#include <utility_kernels_pose.h>

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
    throw std::runtime_error("Usage: ./cmd_line_render <input.h5> <output.h5>");

  // create files
  util::HDF5File in_file(argv[1]);
  util::HDF5File out_file(argv[2]);

  std::vector<int> size_t, size_r;
  std::vector<double> data_t, data_r;
  in_file.readArray("t", data_t, size_t);
  in_file.readArray("r", data_r, size_r);

  int n_objects = size_t.at(0);
  if (size_t != size_r)
    throw std::runtime_error("t and r expected to be of equal size");

  std::vector<pose::TranslationRotation3D> poses;

  for (int i = 0; i < n_objects; i++) {
    pose::TranslationRotation3D pose;
    pose.setT(&data_t.at(i * 3));
    pose.setR(&data_r.at(i * 3));
    //    pose.show();
    poses.push_back(pose);
  }

  std::vector<int> size_file_names;
  std::vector<std::string> obj_file_names;
  in_file.readArray("obj_file_names", obj_file_names, size_file_names);

  //  for(auto &it : obj_file_names)
  //    std::cout << it << std::endl;

  int width = fetchScalar<int>(in_file, "width", 640);
  int height = fetchScalar<int>(in_file, "height", 480);

  cv::Mat camera_matrix;
  {
    std::vector<int> size;
    std::vector<double> data;
    in_file.readArray("camera_matrix", data, size);
    if ((size.at(0) != 3) || (size.at(1) != 4))
      throw std::runtime_error("Expecting 4x3 camera_matrix");
    camera_matrix = cv::Mat(3, 4, CV_64FC1, data.data()).clone();
  }

  double fx = camera_matrix.at<double>(0, 0);
  double fy = camera_matrix.at<double>(1, 1);
  double cx = camera_matrix.at<double>(0, 2);
  double cy = camera_matrix.at<double>(1, 2);

  float near_plane = fetchScalar<float>(in_file, "near_plane", .001f);
  float far_plane = fetchScalar<float>(in_file, "far_plane", 3.0f);
  float scale = fetchScalar<float>(in_file, "scale", 1.0f);

  /***********/
  /* PROCESS */
  /***********/

  pose::MultipleRigidModelsOgre models(width, height, fx, fy, cx, cy,
                                       near_plane, far_plane);
  for (int i = 0; i < n_objects; i++)
    models.addModel(obj_file_names.at(i));
  models.render(poses);

  /**********/
  /* OUTPUT */
  /**********/

  std::vector<int> out_size{ height, width };
  std::vector<float> h_out(height * width);

  cudaMemcpyFromArray(h_out.data(), models.getTexture(), 0, 0,
                      height * width * 4, cudaMemcpyDeviceToHost);
  out_file.writeArray("texture", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), models.getNormalX(), 0, 0,
                      height * width * 4, cudaMemcpyDeviceToHost);
  out_file.writeArray("normal_x", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), models.getNormalY(), 0, 0,
                      height * width * 4, cudaMemcpyDeviceToHost);
  out_file.writeArray("normal_y", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), models.getNormalZ(), 0, 0,
                      height * width * 4, cudaMemcpyDeviceToHost);
  out_file.writeArray("normal_z", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), models.getSegmentIND(), 0, 0,
                      height * width * 4, cudaMemcpyDeviceToHost);
  out_file.writeArray("segment_ind", h_out, out_size);

  util::Device1D<float> d_z(height * width);
  pose::convertZbufferToZ(d_z.data(), models.getZBuffer(), width, height, cx,
                          cy, near_plane, far_plane);
  d_z.copyTo(h_out);
  out_file.writeArray("z", h_out, out_size);

  return EXIT_SUCCESS;
}

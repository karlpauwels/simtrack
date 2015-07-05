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
#include <multi_rigid_detector.h>

int main(int argc, char **argv) {

  /*********/
  /* INPUT */
  /*********/

  if (argc < 3)
    throw std::runtime_error(
        "Usage: ./cmd_line_detect_object_pose <input.h5> <output.h5>");

  // create files
  util::HDF5File in_file(argv[1]);

  std::vector<int> image_size;
  std::vector<uint8_t> image_data;

  in_file.readArray("image", image_data, image_size);

  if (image_size.size() != 2)
    throw std::runtime_error("Expecting 2D uint8_t image");

  // default arguments
  interface::MultiRigidDetector::Parameters parameters;
  int device_id = 0;
  if (in_file.checkVariableExists("device_id"))
    device_id = in_file.readScalar<int>("device_id");
  if (in_file.checkVariableExists("num_iter_ransac"))
    parameters.num_iter_ransac_ = in_file.readScalar<int>("num_iter_ransac");
  if (in_file.checkVariableExists("vec_size"))
    parameters.vec_size_ = in_file.readScalar<int>("vec_size");

  std::string obj_file_name = in_file.readScalar<std::string>("obj_file_name");
  std::vector<std::string> obj_filenames{ obj_file_name };

  cv::Mat camera_matrix;
  {
    std::vector<int> size;
    std::vector<double> data;
    in_file.readArray("camera_matrix", data, size);
    if ((size.at(0) != 3) || (size.at(1) != 4))
      throw std::runtime_error("Expecting 4x3 camera_matrix");
    camera_matrix = cv::Mat(3, 4, CV_64FC1, data.data()).clone();
  }

  /***********/
  /* PROCESS */
  /***********/

  int image_width = image_size.at(1);
  int image_height = image_size.at(0);
  cv::Mat image(image_height, image_width, CV_8UC1, image_data.data());

  interface::MultiRigidDetector detector(
      image_width, image_height, camera_matrix, obj_filenames, device_id);

  pose::TranslationRotation3D pose;
  detector.estimatePose(image, 0, pose);

  /**********/
  /* OUTPUT */
  /**********/

  util::HDF5File out_file(argv[2]);

  {
    std::vector<int> size{ 1, 3 };
    std::vector<double> data(3);

    pose.getT((double *)data.data());
    out_file.writeArray("t", data, size);
    pose.getR((double *)data.data());
    out_file.writeArray("r", data, size);
  }

  {
    std::vector<int> size{ 3, 3 };
    std::vector<double> data(9);
    pose.getR_mat((double *)data.data());
    out_file.writeArray("r_mat", data, size);
  }

  return EXIT_SUCCESS;
}

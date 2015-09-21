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
#include <multi_rigid_tracker.h>

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
        "Usage: ./cmd_line_track_multi_object <input.h5> <output.h5>");

  // create files
  util::HDF5File in_file(argv[1]);
  util::HDF5File out_file(argv[2]);

  std::vector<int> images_size;
  std::vector<uint8_t> images_data;

  in_file.readArray("images", images_data, images_size);

  if (images_size.size() != 3)
    throw std::runtime_error("Expecting 2D uint8_t images");

  size_t image_width = images_size.at(2);
  size_t image_height = images_size.at(1);
  size_t n_images = images_size.at(0);

  // default arguments
  bool save_output_images =
      (bool)fetchScalar<int>(in_file, "save_output_images", 0);

  interface::MultiRigidDetector::Parameters detector_parameters;
  vision::D_OpticalAndARFlow::Parameters flow_parameters;
  pose::D_MultipleRigidPoses::Parameters pose_parameters;

  pose_parameters.w_disp_ = 0.0;
//  pose_parameters.w_flow_ = 0.0;
  pose_parameters.w_flow_ = 1.0;

  pose_parameters.check_reliability_ = false;

  std::vector<std::string> tracker_filenames;

  std::vector<int> size;
  in_file.readArray("tracker_filenames", tracker_filenames, size);
  std::vector<std::string> detector_filenames;
  in_file.readArray("detector_filenames", detector_filenames, size);

  if (detector_filenames.size() != tracker_filenames.size())
    throw std::runtime_error(
        "Expecting equal number of tracker and detector filenames");

  int n_objects = detector_filenames.size();

  cv::Mat camera_matrix;
  {
    std::vector<int> size;
    std::vector<double> data;
    in_file.readArray("camera_matrix", data, size);
    if ((size.at(0) != 3) || (size.at(1) != 4))
      throw std::runtime_error("Expecting 4x3 camera_matrix");
    camera_matrix = cv::Mat(3, 4, CV_64FC1, data.data()).clone();
  }

  // allocate output
  std::vector<int> t_r_out_size{ (int)n_images, (int)n_objects, 3 };
  std::vector<double> t_out(3 * n_objects * n_images);
  std::vector<double> r_out(3 * n_objects * n_images);

  std::vector<int> output_images_size;
  std::vector<uint8_t> output_images;

  if (save_output_images) {
    output_images_size = { (int)n_images, (int)image_height, (int)image_width, 3 };
    size_t n_total = n_images * image_height * image_width * 3;
    output_images.resize(n_total);
  }

  /***********/
  /* PROCESS */
  /***********/

  int device_id = 0;
  util::initializeCUDARuntime(device_id);

  interface::MultiRigidDetector detector(
      image_width, image_height, camera_matrix, detector_filenames, device_id);

  std::vector<interface::MultiRigidTracker::ObjectInfo> object_info;
  for (int i = 0; i < n_objects; ++i)
    object_info.push_back(interface::MultiRigidTracker::ObjectInfo(
        "dummy_label", tracker_filenames.at(i)));

  interface::MultiRigidTracker tracker(image_width, image_height, camera_matrix,
                                       object_info, flow_parameters,
                                       pose_parameters);

  for (size_t i = 0; i < n_images; ++i) {
    cv::Mat image(image_height, image_width, CV_8UC1,
                  &images_data.at(image_height * image_width * i));
    int detector_object = i % n_objects;
    pose::TranslationRotation3D detector_pose;
    detector.estimatePose(image, detector_object, detector_pose);
    if (i < n_objects * 2)
      tracker.setRigidDetectorPose(detector_pose, detector_object);
    else {
      pose::TranslationRotation3D dummy_pose;
      tracker.setRigidDetectorPose(dummy_pose, detector_object);
    }

    tracker.updatePoses(image);

    std::vector<pose::TranslationRotation3D> tracker_poses = tracker.getPoses();
    for (int o = 0; o < n_objects; ++o) {
      int IND = 3 * n_objects * i + 3 * o;
      tracker_poses.at(o).getT(&t_out.at(IND));
      tracker_poses.at(o).getR(&r_out.at(IND));
      //      std::cout << i << " " << o << std::endl;
      //      tracker_poses.at(o).showCompact();
    }

    if (save_output_images) {
      cv::Mat output_image = tracker.generateOutputImage(
          interface::MultiRigidTracker::OutputImageType::
              model_appearance_blended);
      cv::Mat image_out(image_height, image_width, CV_8UC3,
                        &output_images.at(image_height * image_width * i * 3));
      cv::cvtColor(output_image, image_out, CV_BGRA2BGR);
    }
  }

  /**********/
  /* OUTPUT */
  /**********/

  out_file.writeArray("t", t_out, t_r_out_size);
  out_file.writeArray("r", r_out, t_r_out_size);

  if (save_output_images) {
    out_file.writeArray("output_images", output_images, output_images_size);
  }

  //  {
  //    std::vector<int> size { 1, 3 };
  //    std::vector<double> data(3);

  //    pose.getT((double *)data.data());
  //    out_file.writeArray("t", data, size);
  //    pose.getR((double *)data.data());
  //    out_file.writeArray("r", data, size);
  //  }

  //  {
  //    std::vector<int> size { 3, 3 };
  //    std::vector<double> data(9);
  //    pose.getR_mat((double *)data.data());
  //    out_file.writeArray("r_mat", data, size);
  //  }

  return EXIT_SUCCESS;
}

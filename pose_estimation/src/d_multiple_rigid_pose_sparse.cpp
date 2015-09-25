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

#include <d_multiple_rigid_pose_sparse.h>
#include <GL/gl.h>
#include <sys/time.h>
#include <cstdio>
#include <stdexcept>
#include <H5Cpp.h>
#include <hdf5_file.h>
#include <utilities.h>
#include <Eigen/Dense>

using namespace cv;

namespace pose {

D_MultipleRigidPoseSparse::D_MultipleRigidPoseSparse(
    int n_cols, int n_rows, float nodal_point_x, float nodal_point_y,
    float focal_length_x, float focal_length_y, int device_id, int vec_size,
    int num_iter_ransac)
    : _running{ true }, _n_objects{ 0 }, _num_iter_ransac{ num_iter_ransac },
      _max_matches{ 50000 }, _DESCRIPTOR_LENGTH{ 128 },
      _siftEngine{ std::unique_ptr<SiftGPU>{ new SiftGPU() } },
      _matcherEngine{ std::unique_ptr<SiftMatchGPU>{ new SiftMatchGPU(
          4096 * vec_size) } } {
  _camera_mat.create(3, 3, CV_32F);
  updateCalibration(n_cols, n_rows, nodal_point_x, nodal_point_y,
                    focal_length_x, focal_length_y);

  char device_id_str[2];
  sprintf(device_id_str, "%d", device_id);

  // allow for full hd with upscaling (2*1920 = 3840)
  // allow for 3K with upscaling (2*2304 = 4608)
  const char *argv_template[] = { "-m",          "-fo",   "-1",    "-s",
                                  "-v",          "0",     "-pack", "-cuda",
                                  device_id_str, "-maxd", "4608" };
  int argc = sizeof(argv_template) / sizeof(char *);

  char *argv[argc];
  for (int i = 0; i < argc; i++)
    argv[i] = strdup(argv_template[i]);

  _siftEngine->ParseParam(argc, argv);

  for (int i = 0; i < argc; i++)
    free(argv[i]);

  if (_siftEngine->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    throw std::runtime_error("D_MultipleRigidPoseSparse::D_"
                             "MultipleRigidPoseSparse: SiftGPU cannot create "
                             "GL contex\n");

  _matcherEngine->VerifyContextGL();
  _match_buffer.resize(_max_matches * 2);
}

void D_MultipleRigidPoseSparse::updateCalibration(int n_cols, int n_rows,
                                                  float nodal_point_x,
                                                  float nodal_point_y,
                                                  float focal_length_x,
                                                  float focal_length_y) {
  _n_cols = n_cols;
  _n_rows = n_rows;

  // probably wrong!
  _camera_mat.at<float>(0, 0) = focal_length_x;
  _camera_mat.at<float>(0, 1) = 0.f;
  _camera_mat.at<float>(0, 2) = nodal_point_x - _n_cols / 2.0;

  _camera_mat.at<float>(1, 0) = 0.f;
  _camera_mat.at<float>(1, 1) = focal_length_y;
  _camera_mat.at<float>(1, 2) = nodal_point_y - _n_rows / 2.0;

  _camera_mat.at<float>(2, 0) = 0.f;
  _camera_mat.at<float>(2, 1) = 0.f;
  _camera_mat.at<float>(2, 2) = 1.f;
}

void D_MultipleRigidPoseSparse::addModel(const char *obj_filename) {
  _n_objects++;

  // Read from HDF5-file
  std::vector<float> descriptors, positions;
std:
  vector<int> d_size, p_size;
  util::HDF5File f(obj_filename);
  f.readArray("descriptors", descriptors, d_size);
  f.readArray("positions", positions, p_size);

  if (d_size.size() != 2)
    throw std::runtime_error("D_MultipleRigidPoseSparse::addModel: descriptors "
                             "field should be 2D\n");
  if (d_size.at(1) != _DESCRIPTOR_LENGTH)
    throw std::runtime_error(std::string(
        "D_MultipleRigidPoseSparse::addModel: descriptors should be " +
        std::to_string(_DESCRIPTOR_LENGTH) + "-dimensional\n"));
  if (p_size.size() != 2)
    throw std::runtime_error(
        "D_MultipleRigidPoseSparse::addModel: positions field should be 2D\n");
  if (p_size.at(1) != 3)
    throw std::runtime_error("D_MultipleRigidPoseSparse::addModel: positions "
                             "should be 3-dimensional\n");
  if (d_size.at(0) != p_size.at(0))
    throw std::runtime_error("D_MultipleRigidPoseSparse::addModel: different "
                             "number of positions and descriptors\n");

  ModelAssets model;
  model.model_size = d_size.at(0);

  for (int i = 0; i < model.model_size; i++) {
    SiftGPU::SiftKeypoint position;
    position.x = positions.at(i * 3);
    position.y = positions.at(i * 3 + 1);
    position.s = positions.at(i * 3 + 2);
    position.o = 0;
    model.positions.push_back(position);
  }

  model.descriptors = descriptors;

  _allModels.push_back(model);
}

void D_MultipleRigidPoseSparse::removeAllModels() {
  _allModels.clear();
  _n_objects = 0;
}

TranslationRotation3D
D_MultipleRigidPoseSparse::estimatePoseSpecificObject(const Mat &image,
                                                      const int object) {
  return estimatePose(image, object);
}

TranslationRotation3D
D_MultipleRigidPoseSparse::estimatePoseRandomObject(const Mat &image,
                                                    int &object) {
  double obj_probabilities[_n_objects];
  for (int i = 0; i < _n_objects; i++)
    obj_probabilities[i] = 1.0 / (double)_n_objects;

  double r = ((double)rand() / double(RAND_MAX));

  object = 0;
  double prob_ulim = obj_probabilities[object];

  while (r > prob_ulim) {
    object++;
    prob_ulim += obj_probabilities[object];
  }

  //  printf("%d -- %2.4f -- %2.4f %2.4f
  // %d!!!!\n",_n_objects,r,obj_probabilities[0],obj_probabilities[1],object);

  return estimatePose(image, object);
}

TranslationRotation3D D_MultipleRigidPoseSparse::estimatePose(const Mat &image,
                                                              int object) {

  TranslationRotation3D currPose;

  if (_running) {

    ModelAssets &model = _allModels.at(object);

    _siftEngine->RunSIFT(_n_cols, _n_rows, image.data, GL_LUMINANCE,
                         GL_UNSIGNED_BYTE);

    int n_image_features = _siftEngine->GetFeatureNum();

    vector<float> img_feature_descriptors(_DESCRIPTOR_LENGTH *
                                          n_image_features);
    vector<SiftGPU::SiftKeypoint> img_positions(n_image_features);

    _siftEngine->GetFeatureVector(img_positions.data(),
                                  img_feature_descriptors.data());

    const int one_d_texture_limit = 134217728;
    if (model.model_size * n_image_features > one_d_texture_limit) {

      // shuffling and downsampling image keypoints
      int max_n_image_features = one_d_texture_limit / model.model_size - 1;

      //      std::cout << "going to match " << model.model_size << " model to "
      //                << n_image_features << " image features - max allowed
      // image = "
      //                << max_n_image_features << std::endl;

      std::vector<int> shuffle_inds(n_image_features);
      std::iota(shuffle_inds.begin(), shuffle_inds.end(), 0);
      std::random_shuffle(shuffle_inds.begin(), shuffle_inds.end());

      Eigen::Map<Eigen::VectorXi> shuffle_inds_eig(shuffle_inds.data(),
                                                   n_image_features);
      Eigen::Map<Eigen::MatrixXf> all_keypoints_eig(
          (float *)img_positions.data(), 4, n_image_features);
      Eigen::Map<Eigen::MatrixXf> all_descriptors_eig(
          img_feature_descriptors.data(), _DESCRIPTOR_LENGTH, n_image_features);

      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> p(
          shuffle_inds_eig);
      all_keypoints_eig = all_keypoints_eig * p;
      all_descriptors_eig = all_descriptors_eig * p;

      n_image_features = max_n_image_features;
    }

    _matcherEngine->SetDescriptors(0, model.model_size,
                                   model.descriptors.data()); // model
    _matcherEngine->SetDescriptors(1, n_image_features,
                                   img_feature_descriptors.data()); // image

    // match and get result
    int num_match = _matcherEngine->GetSiftMatch(
        _max_matches, (int(*)[2])_match_buffer.data());

    // compute pnp
    vector<Point3f> objectPoints;
    vector<Point2f> imagePoints;

    for (int i = 0; i < num_match; i++) {
      SiftGPU::SiftKeypoint &objectKey =
          model.positions[_match_buffer.at(i * 2)];
      SiftGPU::SiftKeypoint &imageKey =
          img_positions[_match_buffer.at(i * 2 + 1)];

      Point2f imagePoint(imageKey.x - _n_cols / 2, imageKey.y - _n_rows / 2);
      Point3f objectPoint(objectKey.x, objectKey.y, objectKey.s);

      objectPoints.push_back(objectPoint);
      imagePoints.push_back(imagePoint);
    }

    const float max_dist = 1.0; // 1.0F

    Mat rvec, tvec;
    vector<int> inliers_cpu;
    if (objectPoints.size() > 4) {
      solvePnPRansac(objectPoints, imagePoints, _camera_mat,
                     Mat::zeros(1, 8, CV_32F), rvec, tvec, false,
                     _num_iter_ransac, max_dist, objectPoints.size(),
                     inliers_cpu, CV_P3P);
      double T[] = { tvec.at<double>(0, 0), tvec.at<double>(0, 1),
                     tvec.at<double>(0, 2) };
      double R[] = { rvec.at<double>(0, 0), rvec.at<double>(0, 1),
                     rvec.at<double>(0, 2) };
      currPose = TranslationRotation3D(T, R);
    }

    // require at least one inlier for the estimate to be valid (apart from the
    // four points used to estimate pose)
    currPose.setValid(currPose.isFinite() && (inliers_cpu.size() >= 5));

    // if (currPose.isValid()) {
    //   currPose.showCompact();
    //   std::cout << "inliers: " << inliers_cpu.size() << "/"
    //             << objectPoints.size() << std::endl << std::endl;
    // }

    //    double pnp_time = timer.read();

    //    std::cout << std::setprecision(2);
    //    std::cout << "extract: " << extract_time << " match: " << match_time
    //              << " pnp: " << pnp_time << std::endl;
  }

  return (currPose);
}
}

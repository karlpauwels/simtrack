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
#include <unordered_map>
#include <robot.h>
#include <ogre_context.h>
#include <ogre_multi_render_target.h>
#include <urdf/model.h>
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

  // Render robot together with objects. Object and camera poses are expressed
  // relative to the root robot frame

  /*********/
  /* INPUT */
  /*********/

  if (argc < 3)
    throw std::runtime_error(
        "Usage: ./cmd_line_render_scene <input.h5> <output.h5>");

  // create files
  util::HDF5File in_file(argv[1]);

  // read robot state
  std::vector<int> size_joint_names;
  std::vector<std::string> joint_names;
  in_file.readArray<std::string>("joint_names", joint_names, size_joint_names);
  std::vector<double> joint_angles;
  in_file.readArray<double>("joint_angles", joint_angles, size_joint_names);
  std::string robot_description =
      in_file.readScalar<std::string>("robot_description");

  // read object(s) state
  std::vector<pose::TranslationRotation3D> object_poses;
  std::vector<std::string> obj_file_names;
  int n_objects = 0;
  if (in_file.checkVariableExists("t_objects") &&
      in_file.checkVariableExists("r_objects") &&
      in_file.checkVariableExists("obj_file_names")) {
    std::vector<int> size_t, size_r;
    std::vector<double> data_t, data_r;
    in_file.readArray("t_objects", data_t, size_t);
    in_file.readArray("r_objects", data_r, size_r);

    n_objects = size_t.at(0);
    if (size_t != size_r)
      throw std::runtime_error(
          "t_objects and r_objects expected to be of equal size");

    for (int i = 0; i < n_objects; i++) {
      pose::TranslationRotation3D pose;
      pose.setT(&data_t.at(i * 3));
      pose.setR(&data_r.at(i * 3));
      //    pose.show();
      object_poses.push_back(pose);
    }

    std::vector<int> size_file_names;
    in_file.readArray("obj_file_names", obj_file_names, size_file_names);
  }

  // read camera pose (in world coordinates)
  pose::TranslationRotation3D camera_pose;
  {
    std::vector<int> size_t, size_r;
    std::vector<double> t_camera, r_camera;
    in_file.readArray("t_camera", t_camera, size_t);
    in_file.readArray("r_camera", r_camera, size_r);
    if (size_t != size_r)
      throw std::runtime_error(
          "t_camera and r_camera expected to be of equal size");

    camera_pose.setT(t_camera.data());
    camera_pose.setR(r_camera.data());
  }

  bool show_robot = (bool)fetchScalar<int>(in_file, "show_robot", 1);

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
  float far_plane = fetchScalar<float>(in_file, "far_plane", 10.0f);
  float zoom_x = 1;
  float zoom_y = 1;

  Ogre::Matrix4 projection_matrix = Ogre::Matrix4::ZERO;
  projection_matrix[0][0] = 2.0 * fx / (double)width * zoom_x;
  projection_matrix[1][1] = 2.0 * fy / (double)height * zoom_y;
  projection_matrix[0][2] = 2.0 * (0.5 - cx / (double)width) * zoom_x;
  projection_matrix[1][2] = 2.0 * (cy / (double)height - 0.5) * zoom_y;
  projection_matrix[2][2] =
      -(far_plane + near_plane) / (far_plane - near_plane);
  projection_matrix[2][3] =
      -2.0 * far_plane * near_plane / (far_plane - near_plane);
  projection_matrix[3][2] = -1;

  /***********/
  /* PROCESS */
  /***********/

  render::OgreContext ogre_context;

  if (show_robot) {
    // configure robot
    urdf::Model robot_model;
    robot_model.initString(robot_description);
    render::Robot robot(robot_model, ogre_context.scene_manager_,
                        n_objects + 1);

    render::Robot::M_NameToAngle joint_state;
    for (int i = 0; i < joint_names.size(); i++)
      joint_state[joint_names.at(i)] = joint_angles.at(i);

    robot.setJointState(joint_state);
  }

  // configure objects (if any)
  std::vector<std::unique_ptr<render::RigidObject> > rigid_objects;

  for (int o = 0; o < n_objects; o++) {
    int segment_ind = o + 1;
    std::string model_resource = "file://" + obj_file_names.at(o);

    auto rigid_object =
        std::unique_ptr<render::RigidObject>{ new render::RigidObject(
            model_resource, ogre_context.scene_manager_, segment_ind) };
    rigid_object->setVisible(true);
    rigid_object->setPose(object_poses.at(o).ogreTranslation(),
                          object_poses.at(o).ogreRotation());
    rigid_objects.push_back(std::move(rigid_object));
  }

  // render
  Ogre::Vector3 camera_position = camera_pose.ogreTranslation();
  Ogre::Quaternion camera_orientation = camera_pose.ogreRotation();

  // convert vision (Z-forward) frame to ogre frame (Z-out)
  camera_orientation =
      camera_orientation *
      Ogre::Quaternion(Ogre::Degree(180), Ogre::Vector3::UNIT_X);

  render::OgreMultiRenderTarget ogre_multi_render_target(
      "scene", width, height, ogre_context.scene_manager_);
  ogre_multi_render_target.updateCamera(camera_position, camera_orientation,
                                        projection_matrix);
  ogre_multi_render_target.render();

  /**********/
  /* OUTPUT */
  /**********/

  util::HDF5File out_file(argv[2]);

  std::vector<cudaArray **> cuda_arrays;
  int n_arrays = 6;
  for (int i = 0; i < n_arrays; i++)
    cuda_arrays.push_back(new cudaArray *);

  std::vector<int> out_size{ height, width };
  std::vector<float> h_out(height * width);

  ogre_multi_render_target.mapCudaArrays(cuda_arrays);

  cudaMemcpyFromArray(h_out.data(), *cuda_arrays.at(5), 0, 0,
                      width * height * sizeof(float), cudaMemcpyDeviceToHost);
  out_file.writeArray("texture", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), *cuda_arrays.at(0), 0, 0,
                      width * height * sizeof(float), cudaMemcpyDeviceToHost);
  out_file.writeArray("normal_x", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), *cuda_arrays.at(1), 0, 0,
                      width * height * sizeof(float), cudaMemcpyDeviceToHost);
  out_file.writeArray("normal_y", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), *cuda_arrays.at(2), 0, 0,
                      width * height * sizeof(float), cudaMemcpyDeviceToHost);
  out_file.writeArray("normal_z", h_out, out_size);

  cudaMemcpyFromArray(h_out.data(), *cuda_arrays.at(4), 0, 0,
                      width * height * sizeof(float), cudaMemcpyDeviceToHost);
  out_file.writeArray("segment_ind", h_out, out_size);

  util::Device1D<float> d_z(height * width);
  pose::convertZbufferToZ(d_z.data(), *cuda_arrays.at(3), width, height, cx, cy,
                          near_plane, far_plane);
  d_z.copyTo(h_out);
  out_file.writeArray("z", h_out, out_size);

  ogre_multi_render_target.unmapCudaArrays();

  return EXIT_SUCCESS;
}

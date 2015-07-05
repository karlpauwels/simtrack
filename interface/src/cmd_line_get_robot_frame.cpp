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
#include <unordered_map>
#include <urdf/model.h>
#include <hdf5_file.h>
#include <kdl_parser/kdl_parser.hpp>
#include <translation_rotation_3d.h>
#include <OgreVector3.h>
#include <OgreQuaternion.h>

typedef std::unordered_map<std::string, double> M_NameToAngle;

std::unordered_map<std::string, KDL::Frame> link_frames_;

void propagateTree(const KDL::SegmentMap::const_iterator segment,
                   const KDL::Frame frame, const M_NameToAngle &joint_state) {
  const std::string &root = GetTreeElementSegment(segment->second).getName();

  auto joint = GetTreeElementSegment(segment->second).getJoint();

  double joint_angle = (joint.getType() == KDL::Joint::None)
                           ? 0.0
                           : joint_state.at(joint.getName());

  KDL::Frame child_frame = frame * (segment->second.segment.pose(joint_angle));
  link_frames_[root] = child_frame;

  const std::vector<KDL::SegmentMap::const_iterator> &children =
      GetTreeElementChildren(segment->second);
  for (unsigned int i = 0; i < children.size(); i++) {
    propagateTree(children[i], child_frame, joint_state);
  }
}

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
        "Usage: ./cmd_line_get_robot_frame <input.h5> <output.h5>");

  // create files
  util::HDF5File in_file(argv[1]);
  util::HDF5File out_file(argv[2]);

  std::vector<int> size_joint_names;
  std::vector<std::string> joint_names;
  in_file.readArray<std::string>("joint_names", joint_names, size_joint_names);
  std::vector<double> joint_angles;
  in_file.readArray<double>("joint_angles", joint_angles, size_joint_names);

  std::string robot_description =
      in_file.readScalar<std::string>("robot_description");
  std::string frame_name = in_file.readScalar<std::string>("frame_name");

  /***********/
  /* PROCESS */
  /***********/

  urdf::Model robot_model;
  robot_model.initString(robot_description);
  KDL::Tree kdl_tree;
  if (!kdl_parser::treeFromUrdfModel(robot_model, kdl_tree))
    throw std::runtime_error("Failed to extract kdl tree from urdf");

  M_NameToAngle joint_state;
  for (int i = 0; i < joint_names.size(); i++)
    joint_state[joint_names.at(i)] = joint_angles.at(i);

  KDL::Frame root_frame;
  propagateTree(kdl_tree.getRootSegment(), root_frame, joint_state);

  KDL::Frame kdl_frame = link_frames_.at(frame_name);
  double x, y, z, w;
  kdl_frame.M.GetQuaternion(x, y, z, w);
  Ogre::Vector3 t(kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z());
  Ogre::Quaternion r(w, x, y, z);
  pose::TranslationRotation3D tr_frame(t, r);

  /**********/
  /* OUTPUT */
  /**********/

  {
    std::vector<int> size{ 1, 3 };
    std::vector<double> data(3);

    tr_frame.getT((double *)data.data());
    out_file.writeArray("t", data, size);
    tr_frame.getR((double *)data.data());
    out_file.writeArray("r", data, size);
  }

  {
    std::vector<int> size{ 3, 3 };
    std::vector<double> data(9);
    tr_frame.getR_mat((double *)data.data());
    out_file.writeArray("r_mat", data, size);
  }

  return EXIT_SUCCESS;
}

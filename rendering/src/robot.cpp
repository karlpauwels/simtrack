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

#include <stdexcept>
#include <robot.h>
#include <kdl_parser/kdl_parser.hpp>

namespace render {

Robot::Robot(const urdf::Model &urdf, Ogre::SceneManager *scene_manager,
             int root_segment_ind) {
  if (!kdl_parser::treeFromUrdfModel(urdf, kdl_tree_))
    throw(std::runtime_error(
        "Robot::Robot: failed to extract kdl tree from urdf"));

  // add the robot links
  int link_ind = 0;
  for (auto &link_it : urdf.links_) {
    if (!link_it.second->visual_array.empty()) {
      // add geometry
      robot_links_.push_back(render::RobotLink(
          *link_it.second.get(), scene_manager, root_segment_ind + link_ind));
      link_ind++;
    }

    // initialize a frame (to ensure consistent state)
    link_frames_[link_it.first] = KDL::Frame::Identity();
  }
}

void Robot::setJointState(const M_NameToAngle &joint_state) {
  // update frames
  KDL::Frame root_frame;
  propagateTree(kdl_tree_.getRootSegment(), root_frame, joint_state);

  // configure robot links
  for (auto &link_it : robot_links_) {
    KDL::Frame frame = link_frames_.at(link_it.getName());
    double x, y, z, w;
    frame.M.GetQuaternion(x, y, z, w);
    Ogre::Vector3 pos(frame.p.x(), frame.p.y(), frame.p.z());
    Ogre::Quaternion orient(w, x, y, z);
    link_it.setPose(pos, orient);
  }
}

void Robot::getFrame(std::string frame, Ogre::Vector3 &position,
                     Ogre::Quaternion &orientation) const {
  KDL::Frame kdl_frame = link_frames_.at(frame);
  double x, y, z, w;
  kdl_frame.M.GetQuaternion(x, y, z, w);
  position = Ogre::Vector3(kdl_frame.p.x(), kdl_frame.p.y(), kdl_frame.p.z());
  orientation = Ogre::Quaternion(w, x, y, z);
}

void Robot::setIncrementalSegmentLabels(int root_segment_ind) {
  int segment_index = root_segment_ind;
  for (auto &it : robot_links_)
    it.setSegmentIndex(segment_index++);
}

void Robot::setFixedSegmentLabels(int segment_index) {
  for (auto &it : robot_links_)
    it.setSegmentIndex(segment_index);
}

void Robot::setVisible(bool visible) {
  for (auto &it : robot_links_)
    it.setVisible(visible);
}

void Robot::setVisible(bool visible, std::vector<std::string> link_names) {
  for (auto &link_it : robot_links_) {
    link_it.setVisible(!visible); // initialize
    for (auto &name_it : link_names) {
      if ( link_it.getName() == name_it )
        link_it.setVisible(visible);
    }
  }
}

void Robot::propagateTree(
    const KDL::SegmentMap::const_iterator segment, const KDL::Frame frame,
    const std::unordered_map<std::string, double> &joint_state) {
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
}

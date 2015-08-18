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

#pragma once

#include <unordered_map>
#include <urdf/model.h>
#include <kdl/tree.hpp>
#include <ogre_context.h>
#include <robot_link.h>

namespace render {

class Robot {

public:
  typedef std::unordered_map<std::string, double> M_NameToAngle;

  Robot(const urdf::Model &urdf, Ogre::SceneManager *scene_manager,
        int root_segment_ind = 1);

  void setJointState(const M_NameToAngle &joint_state);

  void getFrame(std::string frame, Ogre::Vector3 &position,
                Ogre::Quaternion &orientation) const;

  // label the robot parts incrementally starting at root_segment_ind
  void setIncrementalSegmentLabels(int root_segment_ind = 1);

  void setFixedSegmentLabels(int segment_index = 0);

  // enable or disable robot rendering
  void setVisible(bool visible);

private:
  void
  propagateTree(const KDL::SegmentMap::const_iterator segment,
                const KDL::Frame frame,
                const std::unordered_map<std::string, double> &joint_state);

  KDL::Tree kdl_tree_;
  std::vector<render::RobotLink> robot_links_;
  std::unordered_map<std::string, KDL::Frame> link_frames_;
};
}

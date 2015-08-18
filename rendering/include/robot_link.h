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

#include <OgreSceneManager.h>
#include <OgreEntity.h>
#include <urdf/model.h>

namespace render {

class RobotLink {

public:
  RobotLink(const urdf::Link &link, Ogre::SceneManager *scene_manager,
            int segment_ind);

  void setPose(const Ogre::Vector3 &position,
               const Ogre::Quaternion &orientation);

  void setSegmentIndex(int segment_ind);

  void setVisible(bool visible);

  const std::string getName() { return name_; }

private:
  /**
   * @brief createEntityForVisualElement
   * Creates properly offset child node containing the visual mesh
   * @param visual
   * @return
   */
  Ogre::Entity *createEntityForVisualElement(const urdf::Visual &visual);

  Ogre::SceneManager *scene_manager_;
  std::string name_;
  Ogre::SceneNode *visual_node_;
  int segment_ind_;
  std::vector<Ogre::Entity *> entities_;

};
}

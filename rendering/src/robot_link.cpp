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

#include <OgreSubEntity.h>
#include <OgreSubMesh.h>
#include <stdexcept>
#include <mesh_loader.h>
#include <robot_link.h>

namespace render {

RobotLink::RobotLink(const urdf::Link &link, Ogre::SceneManager *scene_manager,
                     int segment_ind)
    : scene_manager_{ scene_manager }, segment_ind_{ segment_ind } {
  visual_node_ = scene_manager_->getRootSceneNode()->createChildSceneNode();
  name_ = link.name;
  for (auto &visual_it : link.visual_array)
    createEntityForVisualElement(*visual_it.get());
  setSegmentIndex(segment_ind);
}

void RobotLink::setPose(const Ogre::Vector3 &position,
                        const Ogre::Quaternion &orientation) {
  visual_node_->setPosition(position);
  visual_node_->setOrientation(orientation);
}

void RobotLink::setSegmentIndex(int segment_ind) {
  Ogre::SceneNode::ChildNodeIterator it = visual_node_->getChildIterator();
  while (it.hasMoreElements())
    static_cast<Ogre::Entity *>(
        static_cast<Ogre::SceneNode *>(it.getNext())->getAttachedObject(0))
        ->getSubEntity(0)
        ->setCustomParameter(1, Ogre::Vector4(segment_ind, 0, 0, 0));
}

void RobotLink::setVisible(bool visible) {
  for(auto &it : entities_)
    it->setVisible(visible);
}

Ogre::Entity *
RobotLink::createEntityForVisualElement(const urdf::Visual &visual) {
  Ogre::SceneNode *offset_node = visual_node_->createChildSceneNode();

  Ogre::Vector3 scale = Ogre::Vector3::UNIT_SCALE;

  Ogre::Vector3 offset_position(visual.origin.position.x,
                                visual.origin.position.y,
                                visual.origin.position.z);
  Ogre::Quaternion offset_orientation(Ogre::Quaternion::IDENTITY);
  offset_orientation =
      offset_orientation *
      Ogre::Quaternion(visual.origin.rotation.w, visual.origin.rotation.x,
                       visual.origin.rotation.y, visual.origin.rotation.z);
  Ogre::Entity *entity = nullptr;

  // fetch geometry

  switch (visual.geometry->type) {
  case urdf::Geometry::SPHERE: {
    entity = scene_manager_->createEntity("rviz_sphere.mesh");
    const urdf::Sphere &sphere =
        static_cast<const urdf::Sphere &>(*visual.geometry.get());
    scale =
        Ogre::Vector3(sphere.radius * 2, sphere.radius * 2, sphere.radius * 2);
    break;
  }
  case urdf::Geometry::BOX: {
    entity = scene_manager_->createEntity("rviz_cube.mesh");
    const urdf::Box &box =
        static_cast<const urdf::Box &>(*visual.geometry.get());
    scale = Ogre::Vector3(box.dim.x, box.dim.y, box.dim.z);
    break;
  }
  case urdf::Geometry::CYLINDER: {
    entity = scene_manager_->createEntity("rviz_cylinder.mesh");
    const urdf::Cylinder &cylinder =
        static_cast<const urdf::Cylinder &>(*visual.geometry.get());
    Ogre::Quaternion rotX;
    rotX.FromAngleAxis(Ogre::Degree(90), Ogre::Vector3::UNIT_X);
    offset_orientation = offset_orientation * rotX;
    scale = Ogre::Vector3(cylinder.radius * 2, cylinder.length,
                          cylinder.radius * 2);
    break;
  }
  case urdf::Geometry::MESH: {
    const urdf::Mesh &mesh =
        static_cast<const urdf::Mesh &>(*visual.geometry.get());
    // next call adds the mesh to the resource group
    render::loadMeshFromResource(mesh.filename);
    entity = scene_manager_->createEntity(mesh.filename);

    scale = Ogre::Vector3(mesh.scale.x, mesh.scale.y, mesh.scale.z);
    break;
  }
  default:
    throw std::runtime_error("Unsupported geometry type");
    break;
  }

  if (entity != nullptr) {
    // set material
    entity->setMaterialName("untextured");

    // configure scene node
    offset_node->attachObject(entity);
    offset_node->setScale(scale);
    offset_node->setPosition(offset_position);
    offset_node->setOrientation(offset_orientation);
  }

  entities_.push_back(entity);

  return (entity);
}
}

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

#include <mesh_loader.h>
#include <OgreEntity.h>
#include <OgreSubEntity.h>
#include <OgreSubMesh.h>
#include <OgreMaterialManager.h>
#include <OgreMeshManager.h>
#include <Eigen/Dense>

#include <rigid_object.h>

namespace render {

RigidObject::RigidObject(std::string file_name,
                         Ogre::SceneManager *scene_manager, int segment_ind)
    : segment_ind_{ segment_ind }, scene_manager_{ scene_manager } {
  Ogre::MeshPtr object_mesh = render::loadMeshFromResource(file_name);
  entity_ = scene_manager_->createEntity(file_name);
  Ogre::MaterialPtr mat =
      Ogre::MaterialManager::getSingleton().getByName("textured");
  std::stringstream ss;
  ss << "textured_" << segment_ind_;
  Ogre::MaterialPtr cloned_mat = mat->clone(ss.str());
  Ogre::Pass *pass = cloned_mat->getTechnique(0)->getPass(0);
  Ogre::TextureUnitState *tex_unit = pass->createTextureUnitState();
  Ogre::MaterialPtr tmp = Ogre::MaterialManager::getSingleton().getByName(
      object_mesh->getSubMesh(0)->getMaterialName());
  tex_unit->setTextureName(tmp->getTechnique(0)
                               ->getPass(0)
                               ->getTextureUnitState(0)
                               ->getTextureName());
  entity_->setMaterial(cloned_mat);

  texture_name_ = tex_unit->getTextureName();
  file_name_ = file_name;

  Ogre::SubEntity *pSub = entity_->getSubEntity(0);
  // mark segment index for vertex shader
  pSub->setCustomParameter(1, Ogre::Vector4(segment_ind_, 0, 0, 0));

  visual_node_ = scene_manager_->getRootSceneNode()->createChildSceneNode();
  visual_node_->attachObject(entity_);

  // store the vertex positions
  extractMeshPositions(object_mesh.get());
  //  std::cout << "extracted " << positions_.size() << " vertices\n";

  // compute the bounding box
  computeBoundingBox();
}

RigidObject::~RigidObject() {
  visual_node_->detachAllObjects();
  scene_manager_->destroyEntity(entity_);
  scene_manager_->destroySceneNode(visual_node_);
  // may still need to remove the materials but most seems to be cleaned up
  Ogre::TextureManager::getSingleton().unload(texture_name_);
  Ogre::TextureManager::getSingleton().remove(texture_name_);
  Ogre::MeshManager::getSingleton().unload(file_name_);
  Ogre::MeshManager::getSingleton().remove(file_name_);
}

void RigidObject::setPose(const Ogre::Vector3 &position,
                          const Ogre::Quaternion &orientation) {
  visual_node_->setPosition(position);
  visual_node_->setOrientation(orientation);
}

void RigidObject::setVisible(bool visible) {
  visual_node_->setVisible(visible);
}

void RigidObject::setSegmentIndex(int segment_ind) {
  segment_ind_ = segment_ind;

  Ogre::SubEntity *pSub = entity_->getSubEntity(0);
  // mark segment index for vertex shader
  pSub->setCustomParameter(1, Ogre::Vector4(segment_ind_, 0, 0, 0));
}

void RigidObject::extractMeshPositions(const Ogre::Mesh *const mesh) {
  bool added_shared = false;
  size_t current_offset = 0;
  size_t shared_offset = 0;
  size_t next_offset = 0;

  n_positions_ = 0;

  // Calculate how many vertices and indices we're going to need
  for (unsigned short i = 0; i < mesh->getNumSubMeshes(); ++i) {
    Ogre::SubMesh *submesh = mesh->getSubMesh(i);
    // We only need to add the shared vertices once
    if (submesh->useSharedVertices) {
      if (!added_shared) {
        n_positions_ += mesh->sharedVertexData->vertexCount;
        added_shared = true;
      }
    } else {
      n_positions_ += submesh->vertexData->vertexCount;
    }
  }

  // Allocate space for the vertices
  positions_.resize(3 * n_positions_);

  added_shared = false;

  // Run through the submeshes again, adding the data into the vector
  for (unsigned short i = 0; i < mesh->getNumSubMeshes(); ++i) {
    Ogre::SubMesh *submesh = mesh->getSubMesh(i);

    Ogre::VertexData *vertex_data = submesh->useSharedVertices
                                        ? mesh->sharedVertexData
                                        : submesh->vertexData;

    if ((!submesh->useSharedVertices) ||
        (submesh->useSharedVertices && !added_shared)) {
      if (submesh->useSharedVertices) {
        added_shared = true;
        shared_offset = current_offset;
      }

      const Ogre::VertexElement *posElem =
          vertex_data->vertexDeclaration->findElementBySemantic(
              Ogre::VES_POSITION);

      Ogre::HardwareVertexBufferSharedPtr vbuf =
          vertex_data->vertexBufferBinding->getBuffer(posElem->getSource());

      unsigned char *vertex = static_cast<unsigned char *>(
          vbuf->lock(Ogre::HardwareBuffer::HBL_READ_ONLY));

      // There is _no_ baseVertexPointerToElement() which takes an Ogre::Real or
      // a double
      //  as second argument. So make it float, to avoid trouble when Ogre::Real
      // will
      //  be compiled/typedefed as double:
      // Ogre::Real* pReal;
      float *pReal;

      for (size_t j = 0; j < vertex_data->vertexCount;
           ++j, vertex += vbuf->getVertexSize()) {
        posElem->baseVertexPointerToElement(vertex, &pReal);
        Ogre::Vector3 pt(pReal[0], pReal[1], pReal[2]);
        //        vertices[current_offset + j] = (orient * (pt * scale)) +
        // position;
        size_t IND = 3 * (current_offset + j);
        positions_.at(IND) = pt.x;
        positions_.at(IND + 1) = pt.y;
        //        positions_.at(IND+2) = -pt.z; // flip z
        positions_.at(IND + 2) = pt.z;
      }

      vbuf->unlock();
      next_offset += vertex_data->vertexCount;
    }
  }
}

void RigidObject::computeBoundingBox() {
  _bounding_box.resize(8 * 3);
  Eigen::Map<const Eigen::MatrixXf> vertices_f(getPositions().data(), 3,
                                               getNPositions());
  Eigen::MatrixXd vertices;
  vertices = vertices_f.cast<double>();

  // subtract vertices mean
  Eigen::Vector3d mean_vertices = vertices.rowwise().mean();
  vertices = vertices - mean_vertices.replicate(1, getNPositions());

  // compute eigenvector covariance matrix
  Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver(vertices *
                                                   vertices.transpose());
  Eigen::MatrixXd real_eigenvectors = eigen_solver.eigenvectors().real();

  // rotate centered vertices with inverse eigenvector matrix
  vertices = real_eigenvectors.transpose() * vertices;

  // compute simple bounding box
  Eigen::Vector3d mn = vertices.rowwise().minCoeff();
  Eigen::Vector3d mx = vertices.rowwise().maxCoeff();
  Eigen::Matrix<double, 3, 8> bounding_box;
  bounding_box << mn(0), mn(0), mn(0), mn(0), mx(0), mx(0), mx(0), mx(0), mn(1),
      mn(1), mx(1), mx(1), mn(1), mn(1), mx(1), mx(1), mn(2), mx(2), mn(2),
      mx(2), mn(2), mx(2), mn(2), mx(2);

  // rotate and translate bounding box back to original position
  Eigen::Matrix3d rot_back = real_eigenvectors;
  Eigen::Translation<double, 3> tra_back(mean_vertices);
  Eigen::Transform<double, 3, Eigen::Affine> t = tra_back * rot_back;
  bounding_box = t * bounding_box;

  // convert to float
  Eigen::Map<Eigen::MatrixXf> bounding_box_f(_bounding_box.data(), 3, 8);

  bounding_box_f = bounding_box.cast<float>();
}
}

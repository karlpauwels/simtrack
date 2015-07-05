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
#include <OgreCamera.h>
#include <cuda_runtime.h>

namespace render {

class OgreMultiRenderTarget {

public:
  OgreMultiRenderTarget(std::string name, int width, int height,
                        Ogre::SceneManager *scene_manager);
  ~OgreMultiRenderTarget();

  void updateCamera(const Ogre::Vector3 &camera_position,
                    const Ogre::Quaternion &camera_orientation,
                    const Ogre::Matrix4 &projection_matrix);

  void render();

  enum class ArrayType {
    normal_x,
    normal_y,
    normal_z,
    z_buffer,
    segment_ind,
    texture
  };

  void mapCudaArrays(std::vector<cudaArray **> cuda_arrays);
  void unmapCudaArrays();

  const std::string name_;

private:
  const int width_;
  const int height_;
  const int n_rtt_textures_;

  Ogre::SceneManager *scene_manager_;
  Ogre::Camera *camera_;
  Ogre::MultiRenderTarget *multi_render_target_;
  std::vector<cudaGraphicsResource *> cuda_resources_;
};
}

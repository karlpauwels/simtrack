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

#include <vector>
#include <OgreVector3.h>
#include <OgreQuaternion.h>
#undef Success
#include <Eigen/Dense>

namespace pose {

class TranslationRotation3D {

public:
  /*! \brief Init to T = [ 0 0 0 ] and R = [ 0 0 0 ] */
  TranslationRotation3D(bool valid = true);

  TranslationRotation3D(const double *T_in, const double *R_in);

  /*!
   * \brief Initialize from 6 element array containing translation and rotation
   * vector
   * Implemented for float and double array
   * \param TR_in
   */
  template <typename Type> TranslationRotation3D(const Type TR_in[6]);

  TranslationRotation3D(const Ogre::Vector3 &ogre_translation,
                        const Ogre::Quaternion &ogre_rotation);

  bool operator==(const TranslationRotation3D &op) const;
  bool operator!=(const TranslationRotation3D &op) const;

  // boolean AND applied to validity
  TranslationRotation3D &operator*=(const TranslationRotation3D &rhs);

  /*! \brief Returns squared norm of translation vector */
  double normT2() const;

  /*! \brief Returns squared norm of rotation vector */
  double normR2() const;

  // convert from left- to right-handed coordinate frame (invertible)
  TranslationRotation3D changeHandedness() const;

  TranslationRotation3D rotateX180() const;

  TranslationRotation3D inverseTransform() const;

  Ogre::Vector3 ogreTranslation() const;

  Ogre::Quaternion ogreRotation() const;

  Eigen::MatrixXd adjoint() const;

  void createGLModelMatrix(float *M_out) const;

  template <typename Type> void getT(Type *T_out) const {
    for (int i = 0; i < 3; i++)
      T_out[i] = static_cast<Type>(T_[i]);
  }

  template <typename Type> void getR(Type *R_out) const {
    for (int i = 0; i < 3; i++)
      R_out[i] = static_cast<Type>(R_[i]);
  }

  template <typename Type> void getR_mat(Type *R_mat_out) const {
    for (int i = 0; i < 9; i++)
      R_mat_out[i] = static_cast<Type>(R_mat_[i]);
  }

  void getQuaternion(double &x, double &y, double &z, double &w) const;

  /*!
   * \brief return homogeneous transformation matrix (row-major)
   * \param F_out
   */
  void getF(double *F_out) const;

  bool isValid() const { return (valid_); }

  bool isFinite() const;

  void getEuler(double &Ex, double &Ey, double &Ez) const;

  void setT(const double *T_in);
  void setR(const double *R_in);
  void setR_mat(double *R_mat_in);

  /**
   * @brief setF
   * Set from 4x4 homogeneous transformation matrix (row-major)
   * @param F_in
   */
  void setF(const std::vector<double> &F_in);

  void setValid(bool valid) { valid_ = valid; }

  void translateX(double Tx);
  void translateY(double Ty);
  void translateZ(double Tz);

  void rotateX(double angle_deg);
  void rotateY(double angle_deg);
  void rotateZ(double angle_deg);

  void show() const;
  void showCompact() const;

private:
  /*! \brief Update R based on R_mat */
  void updateR();

  /*! \brief Update R_mat based on R */
  void updateR_mat();

  double T_[3]; // translation
  // axis-angle representation of rotation
  double R_[3]; // rotation angles (in rad)

  double R_mat_[3 * 3]; // rotation matrix

  bool valid_;

  static const double PI_;
};

inline TranslationRotation3D operator*(TranslationRotation3D lhs,
                                       const TranslationRotation3D &rhs) {
  lhs *= rhs;
  return lhs;
}
}

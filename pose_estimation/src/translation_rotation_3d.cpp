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

#include <cmath>
#include <cstdio>
#include <translation_rotation_3d.h>
#undef Success
#include <Eigen/Geometry>

namespace pose {

const double TranslationRotation3D::PI_ = 3.1415926535897931;

TranslationRotation3D::TranslationRotation3D(bool valid) : valid_{ valid } {
  double tmpT[3] = { 0.0, 0.0, 0.0 };
  setT(tmpT);
  double tmpR[3] = { 0.0, 0.0, 0.0 };
  setR(tmpR);
}

TranslationRotation3D::TranslationRotation3D(const double *T_in,
                                             const double *R_in)
    : valid_{ true } {
  setT(T_in);
  setR(R_in);
}

template <typename Type>
TranslationRotation3D::TranslationRotation3D(const Type TR_in[6])
    : valid_{ true } {
  double T_in[3], R_in[3];
  for (int i = 0; i < 3; i++) {
    T_in[i] = static_cast<double>(TR_in[i]);
    R_in[i] = static_cast<double>(TR_in[i + 3]);
  }
  setT(T_in);
  setR(R_in);
}
template TranslationRotation3D::TranslationRotation3D<float>(
    const float TR_in[6]);
template TranslationRotation3D::TranslationRotation3D<double>(
    const double TR_in[6]);

TranslationRotation3D::TranslationRotation3D(
    const Ogre::Vector3 &ogre_translation,
    const Ogre::Quaternion &ogre_rotation)
    : valid_{ true } {
  double tmpT[3] = { ogre_translation.x, ogre_translation.y,
                     ogre_translation.z };
  double tmpR_mat[9];

  Eigen::Quaterniond q_eigen(ogre_rotation.w, ogre_rotation.x, ogre_rotation.y,
                             ogre_rotation.z);

  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_eig(tmpR_mat);
  rot_eig = q_eigen.toRotationMatrix();

  setT(tmpT);
  setR_mat(tmpR_mat);
}

bool TranslationRotation3D::operator==(const TranslationRotation3D &op) const {
  bool equal = true;
  for (int i = 0; i < 3; i++) {
    equal = (equal && (op.T_[i] == T_[i]));
    equal = (equal && (op.R_[i] == R_[i]));
  }
  equal = (equal && (op.valid_ == valid_));
  return (equal);
}

bool TranslationRotation3D::operator!=(const TranslationRotation3D &op) const {
  return (!(*this == op));
}

TranslationRotation3D &TranslationRotation3D::
operator*=(const TranslationRotation3D &rhs) {
  // Fl = Fl*Fr;
  Eigen::Map<Eigen::Vector3d> tra_left(T_);
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_left(R_mat_);
  double T_rhs[3];
  rhs.getT(T_rhs);
  Eigen::Map<Eigen::Vector3d> tra_right(T_rhs);
  double R_mat_rhs[9];
  rhs.getR_mat(R_mat_rhs);
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_right(
      R_mat_rhs);

  tra_left = rot_left * tra_right + tra_left;
  rot_left *= rot_right;
  updateR();

  // convert back to rotation matrix to increase numerical stability
  updateR_mat();

  // apply logical AND to validity
  setValid(isValid() && rhs.isValid());

  return *this;
}

double TranslationRotation3D::normT2() const {
  return (T_[0] * T_[0] + T_[1] * T_[1] + T_[2] * T_[2]);
}

double TranslationRotation3D::normR2() const {
  return (R_[0] * R_[0] + R_[1] * R_[1] + R_[2] * R_[2]);
}

TranslationRotation3D TranslationRotation3D::changeHandedness() const {

  // Eigen conversion
  // ----------------

  //  Eigen::Vector3d t(t_buf[0], t_buf[1], t_buf[2]);
  //  Eigen::Matrix3d Rl;

  //  Rl << r_buf[0], r_buf[1], r_buf[2],
  //      r_buf[3], r_buf[4], r_buf[5],
  //      r_buf[6], r_buf[7], r_buf[8];

  //  // transform translation to right-handed system (flip-y)
  //  t(1) = -t(1);

  //  // transform rotation to right-handed system
  //  // flip-y, followed by 180 degree rotation around x-axis to convert ogre
  // (z-out)
  //  // to vision (z-forward)
  //  Eigen::Matrix3d Sy = Eigen::Matrix<double, 3, 3>::Identity();
  //  Sy(1,1) = -1.0;
  //  Eigen::Matrix3d Rx_180 = Eigen::Matrix<double, 3, 3>::Identity();
  //  Rx_180(1,1) = -1.0;
  //  Rx_180(2,2) = -1.0;
  //  Eigen::Matrix3d R = Sy * Rl * Sy * Rx_180;

  // direct conversion
  // -----------------

  // flip ty
  double T_out[]{ T_[0], -T_[1], T_[2] };

  // (flip row 2 and then flip column 3)
  double R_out[]{ R_mat_[0], R_mat_[1], -R_mat_[2], -R_mat_[3], -R_mat_[4],
                  R_mat_[5], R_mat_[6], R_mat_[7],  -R_mat_[8] };

  TranslationRotation3D TR_out;
  TR_out.setT(T_out);
  TR_out.setR_mat(R_out);
  TR_out.setValid(valid_);

  return TR_out;
}

TranslationRotation3D TranslationRotation3D::rotateX180() const {
  double T_out[]{ T_[0], T_[1], T_[2] };
  double R_out[]{ R_mat_[0],  -R_mat_[1], -R_mat_[2], R_mat_[3], -R_mat_[4],
                  -R_mat_[5], R_mat_[6],  -R_mat_[7], -R_mat_[8] };

  TranslationRotation3D TR_out;
  TR_out.setT(T_out);
  TR_out.setR_mat(R_out);
  TR_out.setValid(valid_);

  return TR_out;
}

TranslationRotation3D TranslationRotation3D::inverseTransform() const {

  Eigen::Map<const Eigen::Vector3d> tra(T_);
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot(R_mat_);

  double rot_inv_ptr[9];
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_inv(
      rot_inv_ptr);

  rot_inv = rot.transpose();
  double tra_inv_ptr[3];
  Eigen::Map<Eigen::Vector3d> tra_inv(tra_inv_ptr);
  tra_inv = -rot_inv * tra;

  TranslationRotation3D pose_inv;
  pose_inv.setT(tra_inv_ptr);
  pose_inv.setR_mat(rot_inv_ptr);
  pose_inv.setValid(valid_);

  return (pose_inv);
}

Ogre::Vector3 TranslationRotation3D::ogreTranslation() const {
  return Ogre::Vector3(T_[0], T_[1], T_[2]);
}

void TranslationRotation3D::getQuaternion(double &x, double &y, double &z,
                                          double &w) const {
  Eigen::Matrix3d R_eigen;
  R_eigen << R_mat_[0], R_mat_[1], R_mat_[2], R_mat_[3], R_mat_[4], R_mat_[5],
      R_mat_[6], R_mat_[7], R_mat_[8];

  Eigen::Quaterniond q_eigen;
  q_eigen = R_eigen;

  x = q_eigen.x();
  y = q_eigen.y();
  z = q_eigen.z();
  w = q_eigen.w();
}

Ogre::Quaternion TranslationRotation3D::ogreRotation() const {
  double x, y, z, w;
  getQuaternion(x, y, z, w);
  return Ogre::Quaternion(w, x, y, z);
}

Eigen::MatrixXd TranslationRotation3D::adjoint() const {
  double F_ptr[4 * 4];
  Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> > F(F_ptr);
  getF(F_ptr);
  Eigen::MatrixXd adj(6, 6);
  Eigen::Vector3d T = F.topRightCorner<3, 1>();
  Eigen::Matrix3d R = F.topLeftCorner<3, 3>();
  Eigen::Matrix3d skewT;
  skewT << 0.0, -T(2), T(1), T(2), 0.0, -T(0), -T(1), T(0), 0.0;
  adj << R, skewT *R, Eigen::Matrix3d::Zero(), R;
  return adj;
}

void TranslationRotation3D::createGLModelMatrix(float *M_out) const {
  double TGL[3], RGL[3];
  getT(TGL);
  getR(RGL);

  TGL[2] = -TGL[2];
  RGL[2] = -RGL[2];

  TranslationRotation3D TR(TGL, RGL);

  double R_matGL[9];
  TR.getR_mat(R_matGL);

  M_out[0] = R_matGL[0];
  M_out[1] = R_matGL[1];
  M_out[2] = R_matGL[2];
  M_out[3] = 0.0;
  M_out[4] = R_matGL[3];
  M_out[5] = R_matGL[4];
  M_out[6] = R_matGL[5];
  M_out[7] = 0.0;
  M_out[8] = R_matGL[6];
  M_out[9] = R_matGL[7];
  M_out[10] = R_matGL[8];
  M_out[11] = 0.0;
  M_out[12] = TGL[0];
  M_out[13] = TGL[1];
  M_out[14] = TGL[2];
  M_out[15] = 1.0;
}

bool TranslationRotation3D::isFinite() const {
  bool res = true;
  for (int i = 0; i < 3; i++) {
    res = res && std::isfinite(T_[i]);
    res = res && std::isfinite(R_[i]);
  }

  return res;
}

void TranslationRotation3D::getEuler(double &Ex, double &Ey, double &Ez) const {
  //  [ 0 1 2
  //    3 4 5
  //    6 7 8 ]
  // http://nghiaho.com/?page_id=846

  double r32 = R_mat_[7];
  double r33 = R_mat_[8];
  double r31 = R_mat_[6];
  double r21 = R_mat_[3];
  double r11 = R_mat_[0];

  Ex = atan2(r32, r33);
  Ey = atan2(-r31, sqrt(r32 * r32 + r33 * r33));
  Ez = atan2(r21, r11);
}

void TranslationRotation3D::getF(double *F_out) const {
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_mat(
      R_mat_);
  Eigen::Map<const Eigen::Vector3d> trans(T_);
  Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor> > hom(F_out);
  hom << rot_mat, trans, 0.0, 0.0, 0.0, 1.0;
}

void TranslationRotation3D::setT(const double *T_in) {
  for (int i = 0; i < 3; i++)
    T_[i] = T_in[i];
}

void TranslationRotation3D::setR(const double *R_in) {
  for (int i = 0; i < 3; i++)
    R_[i] = R_in[i];
  updateR_mat();
}

void TranslationRotation3D::setR_mat(double *R_mat_in) {
  for (int i = 0; i < 9; i++)
    R_mat_[i] = R_mat_in[i];
  updateR();
}

void TranslationRotation3D::setF(const std::vector<double> &F_in) {
  if (F_in.size() != 16)
    throw std::runtime_error(
        "TranslationRotation3D::setF: F_in requires 16 elements");

  if ((F_in.at(12) != 0.0) || (F_in.at(13) != 0.0) || (F_in.at(14) != 0.0) ||
      (F_in.at(15) != 1.0))
    throw std::runtime_error(
        "TranslationRotation3D::setF: bottom row of F_in should be [0 0 0 1]");

  Eigen::Map<const Eigen::Matrix<double, 4, 4, Eigen::RowMajor> > F_in_eig(
      F_in.data());

  Eigen::Transform<double, 3, Eigen::Affine> F;
  F = F_in_eig;

  double tmpT[3];
  Eigen::Map<Eigen::Vector3d> tra_eig(tmpT);
  tra_eig = F.translation();

  double tmpR_mat[9];
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_eig(tmpR_mat);
  rot_eig = F.rotation();

  setT(tmpT);
  setR_mat(tmpR_mat);
  updateR_mat(); // for stability
}

void TranslationRotation3D::translateX(double Tx) { T_[0] += Tx; }

void TranslationRotation3D::translateY(double Ty) { T_[1] += Ty; }

void TranslationRotation3D::translateZ(double Tz) { T_[2] += Tz; }

void TranslationRotation3D::rotateX(double angle_deg) {
  double angle_rad = angle_deg * PI_ / 180.0;
  double c = cos(angle_rad);
  double s = sin(angle_rad);
  Eigen::Matrix3d Rx;
  Rx << 1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c;
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot(R_mat_);
  rot *= Rx;
  updateR();
  updateR_mat(); // for stability
}

void TranslationRotation3D::rotateY(double angle_deg) {
  double angle_rad = angle_deg * PI_ / 180.0;
  double c = cos(angle_rad);
  double s = sin(angle_rad);
  Eigen::Matrix3d Rx;
  Rx << c, 0.0, s, 0.0, 1.0, 0.0, -s, 0, c;
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot(R_mat_);
  rot *= Rx;
  updateR();
  updateR_mat(); // for stability
}

void TranslationRotation3D::rotateZ(double angle_deg) {
  double angle_rad = angle_deg * PI_ / 180.0;
  double c = cos(angle_rad);
  double s = sin(angle_rad);
  Eigen::Matrix3d Rx;
  Rx << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot(R_mat_);
  rot *= Rx;
  updateR();
  updateR_mat(); // for stability
}

void TranslationRotation3D::show() const {
  printf("T =     %+2.6f %+2.6f %+2.6f\n", T_[0], T_[1], T_[2]);
  printf("R =     %+2.6f %+2.6f %+2.6f\n", R_[0], R_[1], R_[2]);
  printf("R_mat = %+2.6f %+2.6f %+2.6f\n", R_mat_[0], R_mat_[1], R_mat_[2]);
  printf("        %+2.6f %+2.6f %+2.6f\n", R_mat_[3], R_mat_[4], R_mat_[5]);
  printf("        %+2.6f %+2.6f %+2.6f\n", R_mat_[6], R_mat_[7], R_mat_[8]);
}

void TranslationRotation3D::showCompact() const {
  printf("T = %+2.6f %+2.6f %+2.6f | R = %+2.6f %+2.6f %+2.6f ", T_[0], T_[1],
         T_[2], R_[0], R_[1], R_[2]);
  if (isValid())
    printf("valid\n");
  else
    printf("invalid\n");
}

void TranslationRotation3D::updateR() {
  Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_mat(
      R_mat_);
  Eigen::Map<Eigen::Vector3d> rot_axis_angle(R_);
  Eigen::AngleAxis<double> tmp(rot_mat);
  rot_axis_angle = tmp.angle() * tmp.axis();
}

void TranslationRotation3D::updateR_mat() {

  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > rot_mat(R_mat_);
  Eigen::Map<const Eigen::Vector3d> rot_axis_angle(R_);

  double angle = rot_axis_angle.norm();

  if (angle < 1e-15) {
    // identity matrix
    rot_mat = Eigen::Matrix<double, 3, 3>::Identity();
  } else {
    rot_mat = Eigen::AngleAxis<double>(angle, rot_axis_angle / angle)
                  .toRotationMatrix();
  }
}
}

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

#include <iostream>
#include <iomanip>
#include <normal_equations.h>
#undef Success
#include <Eigen/Dense>

namespace pose {

NormalEquations::NormalEquations() {
  _A.resize(36, 0);
  _B.resize(6, 0);
  _dTdR.resize(6, 0);
}

void NormalEquations::reset() {
  std::fill(_A.begin(), _A.end(), 0);
  std::fill(_B.begin(), _B.end(), 0);
  std::fill(_dTdR.begin(), _dTdR.end(), 0);
}

double NormalEquations::squaredNormDeltaT() const {
  return (_dTdR.at(0) * _dTdR.at(0) + _dTdR.at(1) * _dTdR.at(1) +
          _dTdR.at(1) * _dTdR.at(1));
}

void NormalEquations::compose(const float *CO, const float *CD) {

  _A[0] = CO[0];
  _A[1] = 0.0;
  _A[2] = CO[1];
  _A[3] = CO[2];
  _A[4] = CO[3];
  _A[5] = CO[4];
  _A[6] = 0.0;
  _A[7] = CO[0];
  _A[8] = CO[5];
  _A[9] = CO[6];
  _A[10] = -CO[2];
  _A[11] = CO[7];
  _A[12] = CO[1];
  _A[13] = CO[5];
  _A[14] = CO[8];
  _A[15] = CO[9];
  _A[16] = CO[10];
  _A[17] = 0.0;
  _A[18] = CO[2];
  _A[19] = CO[6];
  _A[20] = CO[9];
  _A[21] = CO[11];
  _A[22] = CO[12];
  _A[23] = CO[13];
  _A[24] = CO[3];
  _A[25] = -CO[2];
  _A[26] = CO[10];
  _A[27] = CO[12];
  _A[28] = CO[14];
  _A[29] = CO[15];
  _A[30] = CO[4];
  _A[31] = CO[7];
  _A[32] = 0.0;
  _A[33] = CO[13];
  _A[34] = CO[15];
  _A[35] = CO[16];

  _A[0] += CD[0];
  _A[1] += CD[1];
  _A[2] += CD[2];
  _A[3] += CD[3];
  _A[4] += CD[4];
  _A[5] += CD[5];
  _A[6] += CD[1];
  _A[7] += CD[6];
  _A[8] += CD[7];
  _A[9] += CD[8];
  _A[10] += CD[9];
  _A[11] += CD[10];
  _A[12] += CD[2];
  _A[13] += CD[7];
  _A[14] += CD[11];
  _A[15] += CD[12];
  _A[16] += CD[13];
  _A[17] += CD[14];
  _A[18] += CD[3];
  _A[19] += CD[8];
  _A[20] += CD[12];
  _A[21] += CD[15];
  _A[22] += CD[16];
  _A[23] += CD[17];
  _A[24] += CD[4];
  _A[25] += CD[9];
  _A[26] += CD[13];
  _A[27] += CD[16];
  _A[28] += CD[18];
  _A[29] += CD[19];
  _A[30] += CD[5];
  _A[31] += CD[10];
  _A[32] += CD[14];
  _A[33] += CD[17];
  _A[34] += CD[19];
  _A[35] += CD[20];

  for (int i = 0; i < 6; i++)
    _B[i] = CO[17 + i] + CD[21 + i];
}

void NormalEquations::solve(float *dTdR) {

  Eigen::Map<Eigen::Matrix<double, 6, 6> > A(_A.data());
  Eigen::Map<Eigen::Matrix<double, 6, 1> > B(_B.data());
  Eigen::Map<Eigen::Matrix<double, 6, 1> > double_dTdR(_dTdR.data());

  double_dTdR = A.ldlt().solve(B);

  Eigen::Map<Eigen::Matrix<float, 6, 1> > float_dTdR(dTdR);
  float_dTdR = double_dTdR.cast<float>();
}

void NormalEquations::preCondition() {
  for (auto &it : _A)
    it *= 1.0e-7;
}

void NormalEquations::show() const {
  for (int row = 0; row < 6; row++) {
    std::cout << std::scientific;
    std::cout.precision(3);
    for (int col = 0; col < 6; col++)
      std::cout << std::setw(10) << _A.at(row * 6 + col) << " ";
    std::cout << std::fixed;
    std::cout.precision(6);
    std::cout << ((row == 2) ? "X " : "  ") << std::setw(9) << _dTdR.at(row);
    std::cout << std::scientific;
    std::cout.precision(3);
    std::cout << ((row == 2) ? " = " : "   ") << std::setw(10) << _B.at(row)
              << std::endl;
  }
}
}

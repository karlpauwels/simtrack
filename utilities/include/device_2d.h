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
#include <memory>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

namespace util {

template <class Type> class Device2D {
public:
  // allocates pitchlinear memory to hold width x height elements
  Device2D(int width, int height);
  ~Device2D();

  // remove the rest (rule of five)
  Device2D(const Device2D &) = delete;
  Device2D(Device2D &&) = delete;
  Device2D &operator=(Device2D) = delete;
  Device2D &operator=(Device2D &&) = delete;

  // unique_ptr factory
  typedef std::unique_ptr<Device2D> Ptr;

  static Ptr make_unique(int width, int height) {
    return Ptr{ new Device2D<Type>(width, height) };
  }

  // copy to pre-allocated host vector of correct size and type
  void copyTo(std::vector<Type> &target) const;

  // copy from host vector of correct size and type
  void copyFrom(const std::vector<Type> &source);

  // copy from Device2D of correct size and type
  void copyFrom(const Device2D<Type> &source);

  size_t pitch() const { return (pitch_); }

  // returns raw pointer to data
  Type *data() const { return (device_memory_); }

  const int width_;
  const int height_;

private:
  std::runtime_error cudaException(std::string functionName,
                                   cudaError_t error) const;

  Type *device_memory_;
  const size_t element_size_;
  size_t pitch_;
};

template <class Type>
Device2D<Type>::Device2D(int width, int height)
    : width_(width), height_(height), element_size_(sizeof(Type)) {
  cudaError_t error = cudaMallocPitch((void **)&device_memory_, &pitch_,
                                      width_ * element_size_, height_);
  if (error != cudaSuccess)
    throw cudaException("Device2D::Device2D: ", error);
}

template <class Type> Device2D<Type>::~Device2D() {
  //  std::cout << "running Device2D destructor\n";
  cudaFree(device_memory_);
}

template <class Type>
void Device2D<Type>::copyTo(std::vector<Type> &target) const {
  if (target.size() != width_ * height_)
    throw std::runtime_error(
        std::string("Device2D::copyTo: target vector of incorrect size\n"));

  size_t s_pitch = width_ * element_size_; // host memory is linear

  cudaError_t error =
      cudaMemcpy2D(target.data(), s_pitch, device_memory_, pitch_,
                   width_ * element_size_, height_, cudaMemcpyDeviceToHost);

  if (error != cudaSuccess)
    throw cudaException("Device2D::copyTo: ", error);
}

template <class Type>
void Device2D<Type>::copyFrom(const std::vector<Type> &source) {
  if (source.size() != width_ * height_)
    throw std::runtime_error(
        std::string("Device2D::copyFrom: source vector of incorrect size\n"));

  size_t s_pitch = width_ * element_size_; // host memory is linear

  cudaError_t error =
      cudaMemcpy2D(device_memory_, pitch_, source.data(), s_pitch,
                   width_ * element_size_, height_, cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
    throw cudaException("Device2D::copyFrom: ", error);
}

template <class Type>
void Device2D<Type>::copyFrom(const Device2D<Type> &source) {
  if ((source.width_ != width_) || (source.height_ != height_))
    throw std::runtime_error(
        std::string("Device2D::copyFrom: source Device2D of incorrect size\n"));

  cudaError_t error =
      cudaMemcpy2D(device_memory_, pitch_, source.data(), source.pitch(),
                   width_ * element_size_, height_, cudaMemcpyDeviceToDevice);

  if (error != cudaSuccess)
    throw cudaException("Device2D::copyFrom: ", error);
}

template <class Type>
std::runtime_error Device2D<Type>::cudaException(std::string functionName,
                                                 cudaError_t error) const {
  return (std::runtime_error(functionName +
                             std::string(cudaGetErrorString(error))));
}
}

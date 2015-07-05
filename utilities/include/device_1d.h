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

template <class Type> class Device1D {
public:
  Device1D(int size);
  ~Device1D();

  // remove the rest (rule of five)
  Device1D(const Device1D &) = delete;
  Device1D(Device1D &&) = delete;
  Device1D &operator=(Device1D) = delete;
  Device1D &operator=(Device1D &&) = delete;

  // unique_ptr factory
  typedef std::unique_ptr<Device1D> Ptr;

  static Ptr make_unique(int size) {
    return Ptr{ new Device1D<Type>(size) };
  }

  // copy to same type host vector pre-allocated with equal size
  void copyTo(std::vector<Type> &target) const;

  // copy n_elements to same type host vector pre-allocated with sufficient size
  void copyTo(std::vector<Type> &target, int n_elements) const;

  // copy from same type host vector of equal size
  void copyFrom(const std::vector<Type> &source);

  // copy n_elements from same type host vector
  void copyFrom(const std::vector<Type> &source, int n_elements);

  // copy from same type Device1D of equal size
  void copyFrom(const Device1D<Type> &source);

  // copy n_elements from same type Device1D
  void copyFrom(const Device1D<Type> &source, int n_elements);

  // swap storage pointers with Device1D of identical size and type
  void swap(Device1D<Type> &source);

  // returns raw pointer to data
  Type *data() const { return (device_memory_); }

  const int size_;
  const int size_in_bytes_;

private:
  std::runtime_error cudaException(std::string functionName,
                                   cudaError_t error) const;

  Type *device_memory_;
};

template <class Type>
Device1D<Type>::Device1D(int size)
    : size_(size), size_in_bytes_(sizeof(Type) * size) {
  cudaError_t error = cudaMalloc((void **)&device_memory_, size_in_bytes_);
  if (error != cudaSuccess)
    throw cudaException("Device1D::Device1D: ", error);
}

template <class Type> Device1D<Type>::~Device1D() {
  cudaError_t error = cudaFree(device_memory_);
  if (error != cudaSuccess)
    throw cudaException("Device1D::~Device1D: ", error);
}

template <class Type>
void Device1D<Type>::copyTo(std::vector<Type> &target) const {
  if (target.size() != size_)
    throw std::runtime_error(std::string("Device1D::copyTo: unequal size\n"));

  cudaError_t error = cudaMemcpy(target.data(), device_memory_, size_in_bytes_,
                                 cudaMemcpyDeviceToHost);

  if (error != cudaSuccess)
    throw cudaException("Device1D::copyTo: ", error);
}

template <class Type>
void Device1D<Type>::copyTo(std::vector<Type> &target, int n_elements) const {
  if ((size_ < n_elements) || (target.size() < n_elements))
    throw std::runtime_error(std::string("Device1D::copyTo: size error\n"));

  cudaError_t error =
      cudaMemcpy(target.data(), device_memory_, sizeof(Type) * n_elements,
                 cudaMemcpyDeviceToHost);

  if (error != cudaSuccess)
    throw cudaException("Device1D::copyTo: ", error);
}

template <class Type>
void Device1D<Type>::copyFrom(const std::vector<Type> &source) {
  if (source.size() != size_)
    throw std::runtime_error(std::string("Device1D::copyFrom: unequal size\n"));

  cudaError_t error = cudaMemcpy(device_memory_, source.data(), size_in_bytes_,
                                 cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
    throw cudaException("Device1D::copyFrom: ", error);
}

template <class Type>
void Device1D<Type>::copyFrom(const std::vector<Type> &source, int n_elements) {
  if ((size_ < n_elements) || (source.size() < n_elements))
    throw std::runtime_error(std::string("Device1D::copyFrom: size error\n"));

  cudaError_t error =
      cudaMemcpy(device_memory_, source.data(), sizeof(Type) * n_elements,
                 cudaMemcpyHostToDevice);

  if (error != cudaSuccess)
    throw cudaException("Device1D::copyFrom: ", error);
}

template <class Type>
void Device1D<Type>::copyFrom(const Device1D<Type> &source) {
  if (source.size_ != size_)
    throw std::runtime_error(std::string("Device1D::copyFrom: unequal size\n"));

  cudaError_t error = cudaMemcpy(device_memory_, source.data(), size_in_bytes_,
                                 cudaMemcpyDeviceToDevice);

  if (error != cudaSuccess)
    throw cudaException("Device1D::copyFrom: ", error);
}

template <class Type>
void Device1D<Type>::copyFrom(const Device1D<Type> &source, int n_elements) {
  if ((size_ < n_elements) || (source.size_ < n_elements))
    throw std::runtime_error(std::string("Device1D::copyFrom: size error\n"));

  cudaError_t error =
      cudaMemcpy(device_memory_, source.data(), sizeof(Type) * n_elements,
                 cudaMemcpyDeviceToDevice);

  if (error != cudaSuccess)
    throw cudaException("Device1D::copyFrom: ", error);
}

template <class Type> void Device1D<Type>::swap(Device1D<Type> &source) {
  if (source.size_ != size_)
    throw std::runtime_error(
        std::string("Device1D::swap: source vector of incorrect size\n"));
  Type *source_ptr = source.device_memory_;
  source.device_memory_ = device_memory_;
  device_memory_ = source_ptr;
}

template <class Type>
std::runtime_error Device1D<Type>::cudaException(std::string functionName,
                                                 cudaError_t error) const {
  return (std::runtime_error(functionName +
                             std::string(cudaGetErrorString(error))));
}
}

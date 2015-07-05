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

#include <stdexcept>
#include <utilities.h>
#include <device_1d.h>

namespace util {

void initializeCUDARuntime(int device) {
  cudaSetDevice(device);
  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

  // dummy memcpy to init cuda runtime
  util::Device1D<float> d_dummy(1);
  std::vector<float> h_dummy(1);
  d_dummy.copyFrom(h_dummy);

  if (cudaGetLastError() != cudaSuccess)
    throw std::runtime_error(
        std::string("initializeCUDARuntime: CUDA initialization problem\n"));
}

TimerGPU::TimerGPU(cudaStream_t stream) : stream_(stream) {
  cudaEventCreate(&start_);
  cudaEventCreate(&stop_);
  cudaEventRecord(start_, stream);
}

TimerGPU::~TimerGPU() {
  cudaEventDestroy(start_);
  cudaEventDestroy(stop_);
}

float TimerGPU::read() {
  cudaEventRecord(stop_, stream_);
  cudaEventSynchronize(stop_);
  float time;
  cudaEventElapsedTime(&time, start_, stop_);
  return time;
}

void TimerGPU::reset() { cudaEventRecord(start_, stream_); }
}

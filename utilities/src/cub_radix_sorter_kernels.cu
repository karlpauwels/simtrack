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

#include <cub/cub.cuh>
#include <cub_radix_sorter_kernels.h>

namespace util {

template <class Key, class Value>
size_t GetTempStorageSize(int num_items, int begin_bit, int end_bit) {
  size_t temp_storage_bytes;

  Key *d_key_buf = 0;
  Key *d_key_alt_buf = 0;
  Value *d_value_buf = 0;
  Value *d_value_alt_buf = 0;

  cub::DoubleBuffer<Key> d_keys(d_key_buf, d_key_alt_buf);
  cub::DoubleBuffer<Value> d_values(d_value_buf, d_value_alt_buf);

  cub::DeviceRadixSort::SortPairs(NULL, temp_storage_bytes, d_keys, d_values,
                                  num_items, begin_bit, end_bit);
  return (temp_storage_bytes);
}

template <class Key, class Value>
void CubSort(Key *&d_key_sorted, Value *&d_value_sorted, Key *d_key_buf,
             Key *d_key_alt_buf, Value *d_value_buf, Value *d_value_alt_buf,
             int num_items, int begin_bit, int end_bit, void *d_temp_storage,
             size_t temp_storage_bytes, cudaStream_t stream) {
  // Create a set of DoubleBuffers to wrap pairs of device pointers
  cub::DoubleBuffer<Key> d_keys(d_key_buf, d_key_alt_buf);
  cub::DoubleBuffer<Value> d_values(d_value_buf, d_value_alt_buf);
  cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys,
                                  d_values, num_items, begin_bit, end_bit,
                                  stream);

  // Set output pointer to current doublebuffer
  d_key_sorted = d_keys.Current();
  d_value_sorted = d_values.Current();
}

template size_t GetTempStorageSize<unsigned int, unsigned int>(int num_items,
                                                               int begin_bit,
                                                               int end_bit);
template size_t GetTempStorageSize<unsigned int, int>(int num_items,
                                                      int begin_bit,
                                                      int end_bit);
template size_t GetTempStorageSize<int, float>(int num_items, int begin_bit,
                                               int end_bit);

template void CubSort<int, int>(int *&d_key_sorted, int *&d_value_sorted,
                                int *d_key_buf, int *d_key_alt_buf,
                                int *d_value_buf, int *d_value_alt_buf,
                                int num_items, int begin_bit, int end_bit,
                                void *d_temp_storage, size_t temp_storage_bytes,
                                cudaStream_t stream);
template void CubSort<unsigned int, int>(
    unsigned int *&d_key_sorted, int *&d_value_sorted, unsigned int *d_key_buf,
    unsigned int *d_key_alt_buf, int *d_value_buf, int *d_value_alt_buf,
    int num_items, int begin_bit, int end_bit, void *d_temp_storage,
    size_t temp_storage_bytes, cudaStream_t stream);
template void CubSort<unsigned int, unsigned int>(
    unsigned int *&d_key_sorted, unsigned int *&d_value_sorted,
    unsigned int *d_key_buf, unsigned int *d_key_alt_buf,
    unsigned int *d_value_buf, unsigned int *d_value_alt_buf, int num_items,
    int begin_bit, int end_bit, void *d_temp_storage, size_t temp_storage_bytes,
    cudaStream_t stream);
}

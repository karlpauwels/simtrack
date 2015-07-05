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
#include <stdexcept>
#include <device_1d.h>
#include <cub_radix_sorter_kernels.h>

namespace util {

template <class Key, class Value> class CubRadixSorter {
public:
  /**
   * @brief CubRadixSorter
   * @param num_items
   * @param begin_bit
   * @param end_bit
   * @param stream
   */
  CubRadixSorter(int num_items, int begin_bit, int end_bit,
                 cudaStream_t stream = 0);

  /**
   * @brief sort
   * @param d_key: keys to be sorted
   * @param d_value: values to be sorted according to keys
   * @param d_key_buf: key buffer for double-buffering
   * @param d_value_buf: value buffer for double-buffering
   */
  void sort(util::Device1D<Key> &d_key, util::Device1D<Value> &d_value,
            util::Device1D<Key> &d_key_buf, util::Device1D<Value> &d_value_buf);

private:
  const int num_items_;
  const int begin_bit_;
  const int end_bit_;
  const cudaStream_t stream_;
  util::Device1D<char>::Ptr temp_storage_;
};

template <class Key, class Value>
CubRadixSorter<Key, Value>::CubRadixSorter(int num_items, int begin_bit,
                                           int end_bit, cudaStream_t stream)
    : num_items_{ num_items }, begin_bit_{ begin_bit }, end_bit_{ end_bit },
      stream_{ stream } {
  // allocate temporary storage
  size_t temp_storage_bytes =
      GetTempStorageSize<Key, Value>(num_items_, begin_bit_, end_bit_);
  temp_storage_ =
      util::Device1D<char>::Ptr{ new util::Device1D<char>(temp_storage_bytes) };
}

template <class Key, class Value>
void CubRadixSorter<Key, Value>::sort(util::Device1D<Key> &d_key,
                                      util::Device1D<Value> &d_value,
                                      util::Device1D<Key> &d_key_buf,
                                      util::Device1D<Value> &d_value_buf) {
  if ((d_value.size_ != d_key.size_) || (d_key_buf.size_ != d_key.size_) ||
      (d_value_buf.size_ != d_key.size_))
    throw std::runtime_error(
        "CubRadixSorter:sort: all inputs should be of equal size");

  Key *d_key_sorted;
  Value *d_value_sorted;

  util::CubSort(d_key_sorted, d_value_sorted, d_key.data(), d_key_buf.data(),
                d_value.data(), d_value_buf.data(), d_key.size_, begin_bit_,
                end_bit_, temp_storage_->data(), temp_storage_->size_in_bytes_,
                stream_);

  // swap d_key and d_key_buf if required
  if (d_key_sorted == d_key_buf.data())
    d_key.swap(d_key_buf);

  // swap d_value and d_value_buf if required
  if (d_value_sorted == d_value_buf.data())
    d_value.swap(d_value_buf);
}
}

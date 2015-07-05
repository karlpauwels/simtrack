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
#include <gtest/gtest.h>
#include <hdf5_file.h>
#include <device_1d.h>
#include <device_2d.h>
#include <utilities.h>

TEST(TestHDF5, testWrite) {
  try {
    // remove test file if it still lingers around
    std::string test_filename = "./test.h5";
    if (boost::filesystem::exists(test_filename))
      boost::filesystem::remove(test_filename);
    util::HDF5File file(test_filename);
    int size = 1000;
    std::vector<float> d_values(size);
    std::iota(d_values.begin(), d_values.end(), 0);
    std::vector<int> dims = { 2, 500 };
    file.writeArray("d_values", d_values, dims);
    std::vector<std::string> string_data{
      "first_string", "Second String Is Longer", "3kl", "3kllllll"
    };
    std::vector<int> string_dims{ 2, 2 };
    file.writeArray("text", string_data, string_dims);

    SUCCEED();
    return;
  }
  catch (std::exception &ex) {
    ADD_FAILURE() << ex.what();
  }
}

TEST(TestHDF5, testRead) {
  try {
    std::string test_filename = "./test.h5";
    util::HDF5File file(test_filename);
    std::vector<float> d_values;
    std::vector<int> dims;
    file.readArray("d_values", d_values, dims);
    EXPECT_EQ(2, dims.at(0)) << "Incorrect 0-dim";
    EXPECT_EQ(500, dims.at(1)) << "Incorrect 1-dim";
  }
  catch (std::exception &ex) {
    ADD_FAILURE() << ex.what();
  }
}

TEST(TestCudaContainer, intializeCUDA) {
  cudaError_t err = cudaSetDevice(0);
  ASSERT_EQ(cudaSuccess, err) << cudaGetErrorString(err);
}

TEST(TestCudaContainer, test1D) {
  try {
    int size = 1000;
    std::vector<float> h_data_0(size);
    std::iota(h_data_0.begin(), h_data_0.end(), 0);

    util::Device1D<float> d_1d_0(size);
    EXPECT_EQ(1000 * sizeof(float), d_1d_0.size_in_bytes_) << "Incorrect size";

    util::Device1D<float>::Ptr d_1d_1_ptr;
    //    d_1d_1_ptr = util::Device1D<float>::Ptr(new
    // util::Device1D<float>(size));
    d_1d_1_ptr = util::Device1D<float>::make_unique(size);

    d_1d_0.copyFrom(h_data_0);
    d_1d_1_ptr->copyFrom(d_1d_0);

    std::vector<float> h_data_1(size);
    d_1d_1_ptr->copyTo(h_data_1);
    EXPECT_EQ(h_data_0, h_data_1) << "Unequal after h->d, d->d, d->h copy";
  }
  catch (std::exception &ex) {
    ADD_FAILURE() << ex.what();
  }
}

TEST(TestCudaContainer, test2D) {
  try {
    int width = 2000;
    int height = 20;

    util::Device2D<int> d_2d_0(width, height);

    util::Device2D<int>::Ptr d_2d_1;
    d_2d_1 = util::Device2D<int>::Ptr(new util::Device2D<int>(width, height));

    std::vector<int> h_data_0(width * height);
    std::iota(h_data_0.begin(), h_data_0.end(), 0);

    d_2d_0.copyFrom(h_data_0);
    d_2d_1->copyFrom(d_2d_0);

    std::vector<int> h_data_1(width * height);
    d_2d_1->copyTo(h_data_1);

    EXPECT_EQ(h_data_0, h_data_1) << "Unequal after h->d, d->d, d->h copy";
  }
  catch (std::exception &ex) {
    ADD_FAILURE() << ex.what();
  }
}

TEST(TestTimer, startAndStop) {
  util::TimerGPU timer;
  auto time = timer.read();
}

// Run all the tests that were declared with TEST()
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

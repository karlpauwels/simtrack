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
#include <string>
#include <stdexcept>
#include <H5Cpp.h>
#include <boost/filesystem.hpp>
#include <vector_types.h>

namespace util {

class HDF5File {
public:
  // open for read/write, will create if not exists (existing path required)
  // existing variables with same name cannot be overwritten (hdf5 restriction)
  HDF5File(std::string file_name);

  // remove the rest (rule of five)
  HDF5File(const HDF5File &) = delete;
  HDF5File(HDF5File &&) = delete;
  HDF5File &operator=(HDF5File) = delete;
  HDF5File &operator=(HDF5File &&) = delete;

  // read variable name from file into array, dimensionality stored in dims
  // file datatype will be converted to memory datatype
  template <typename Type>
  void readArray(const std::string &name, std::vector<Type> &array,
                 std::vector<int> &dims);

  // read scalar variable, only succeeds if array is of size 1
  // file datatype will be converted to memory datatype
  template <typename Type> Type readScalar(const std::string &name);

  // write array to file as variable name, with dimensionality specified by dims
  template <typename Type>
  void writeArray(const std::string &name, const std::vector<Type> &array,
                  const std::vector<int> &dims, bool compress = false);

  // read scalar variable, only succeeds if array is of size 1
  template <typename Type>
  void writeScalar(const std::string &name, Type value);

  // checks if variable exists in the file
  bool checkVariableExists(const std::string &name);

private:
  // compilation nullptr error signals that an additional specialization is
  // needed
  // below
  template <typename Type> const H5::PredType typeToH5PredType() {
    return (nullptr);
  }

  std::unique_ptr<H5::H5File> file_;
};

// type conversion specializations

template <> inline const H5::PredType HDF5File::typeToH5PredType<double>() {
  return (H5::PredType::NATIVE_DOUBLE);
}

template <> inline const H5::PredType HDF5File::typeToH5PredType<float>() {
  return (H5::PredType::NATIVE_FLOAT);
}

template <> inline const H5::PredType HDF5File::typeToH5PredType<int>() {
  return (H5::PredType::NATIVE_INT);
}

template <> inline const H5::PredType HDF5File::typeToH5PredType<unsigned int>() {
  return (H5::PredType::NATIVE_UINT);
}

template <> inline const H5::PredType HDF5File::typeToH5PredType<uint8_t>() {
  return (H5::PredType::NATIVE_UINT8);
}

template <> inline const H5::PredType HDF5File::typeToH5PredType<uint16_t>() {
  return (H5::PredType::NATIVE_UINT16);
}

// template<> const H5::PredType HDF5File::typeToH5PredType<unsigned char>() {
//  return(H5::PredType::NATIVE_UCHAR); }

// template<> const H5::PredType HDF5File::typeToH5PredType<short unsigned
// int>() {
//  return(H5::PredType::NATIVE_UINT16); }

inline HDF5File::HDF5File(std::string file_name) {
  H5::Exception::dontPrint();

  boost::filesystem::path path(file_name);

  // check if the file exists
  if (!boost::filesystem::exists(path)) {
    // check if the path exists
    if (!boost::filesystem::exists(path.parent_path())) {
      throw std::runtime_error(std::string("HDF5File::HDF5File: path " +
                                           path.parent_path().string() +
                                           " does not exist "));
    }

    // create the file
    //    std::cout << "creating file " << file_name << std::endl;
    try {
      file_ =
          std::unique_ptr<H5::H5File>(new H5::H5File(file_name, H5F_ACC_TRUNC));
    }
    catch (H5::FileIException error) {
      error.printError();
      throw std::runtime_error(
          std::string("HDF5File::HDF5File: file create problem"));
    }

  } else // file exists, just open it for read/write
  {

    try {
      file_ =
          std::unique_ptr<H5::H5File>(new H5::H5File(file_name, H5F_ACC_RDWR));
    }
    catch (H5::FileIException error) {
      error.printError();
      throw std::runtime_error(
          std::string("HDF5File::HDF5File: file open problem"));
    }
  }
}

inline bool HDF5File::checkVariableExists(const std::string &name) {
  bool dataset_exists = true;
  try {
    H5::DataSet d_set(file_->openDataSet(name));
  }
  catch (H5::FileIException error) {
    dataset_exists = false;
  }
  return (dataset_exists);
}

template <typename Type>
void HDF5File::readArray(const std::string &name, std::vector<Type> &array,
                         std::vector<int> &dims) {
  try {
    H5::DataSet d_set(file_->openDataSet(name));
    H5::DataSpace d_space = d_set.getSpace();
    std::vector<hsize_t> dims_hsize(d_space.getSimpleExtentNdims());
    d_space.getSimpleExtentDims(dims_hsize.data());
    dims.clear();
    std::size_t n_elements = 1;
    for (auto &dim : dims_hsize) {
      dims.push_back(dim);
      n_elements *= dim;
    }
    if (array.size() != n_elements)
      array.resize(n_elements);
    d_set.read(array.data(), typeToH5PredType<Type>());
  }
  catch (H5::FileIException error) {
    error.printError();
    throw std::runtime_error(std::string("HDF5File::readArray: variable " +
                                         name + " doesn't seem to exist"));
  }
}

template <typename Type> Type HDF5File::readScalar(const std::string &name) {
  std::vector<int> dims;
  std::vector<Type> array;
  readArray(name, array, dims);
  if (array.size() != 1)
    throw std::runtime_error(std::string("HDF5File::readScalar: variable " +
                                         name + " is not a scalar"));
  return array.at(0);
}

template <typename Type>
void HDF5File::writeScalar(const std::string &name, Type value) {
  std::vector<Type> array{ value };
  std::vector<int> dims{ 1 };
  bool compress = false;
  writeArray<Type>(name, array, dims, compress);
}

template <typename Type>
void HDF5File::writeArray(const std::string &name,
                          const std::vector<Type> &array,
                          const std::vector<int> &dims, bool compress) {
  // check dataset existence (hdf5 does not allow changes)
  if (checkVariableExists(name))
    throw std::runtime_error(std::string("HDF5File::writeArray: variable " +
                                         name + " already exists"));

  // check consistency of dims
  std::size_t n_elements = 1;
  for (const auto &dim : dims)
    n_elements *= dim;

  if (array.size() != n_elements)
    throw std::runtime_error(
        std::string("HDF5File::writeArray: inconsistent dims"));

  // create dataspace
  std::vector<hsize_t> dims_hsize;
  for (const auto &dim : dims)
    dims_hsize.push_back(dim);
  H5::DataSpace d_space(dims_hsize.size(), dims_hsize.data());

  // create dataset
  H5::DSetCreatPropList ds_creatplist; // create dataset creation prop list
  if (compress) {
    ds_creatplist.setChunk(dims_hsize.size(),
                           dims_hsize.data()); // then modify it for compression
    ds_creatplist.setDeflate(9);
  }

  H5::DataSet d_set(file_->createDataSet(name, typeToH5PredType<Type>(),
                                         d_space, ds_creatplist));

  // write to file
  d_set.write(array.data(), typeToH5PredType<Type>());
}

// float2s can be written as floats by adding dimension 2
template <>
inline void HDF5File::writeArray(const std::string &name,
                                 const std::vector<float2> &array,
                                 const std::vector<int> &dims, bool compress) {
  std::vector<int> dims_float = dims;
  dims_float.push_back(2);
  std::size_t n_elements = 1;
  for (auto &it : dims_float)
    n_elements *= it;
  std::vector<float> array_float;
  array_float.reserve(n_elements);
  for (auto &it : array) {
    array_float.push_back(it.x);
    array_float.push_back(it.y);
  }

  writeArray(name, array_float, dims_float, compress);
}

// uchar4s can be written as uint8_t by adding dimension 4
template <>
inline void HDF5File::writeArray(const std::string &name,
                                 const std::vector<uchar4> &array,
                                 const std::vector<int> &dims, bool compress) {
  std::vector<int> dims_uchar4 = dims;
  dims_uchar4.push_back(4);
  std::size_t n_elements = 1;
  for (auto &it : dims_uchar4)
    n_elements *= it;
  std::vector<uint8_t> array_uchar;
  array_uchar.reserve(n_elements);
  for (auto &it : array) {
    array_uchar.push_back(it.x);
    array_uchar.push_back(it.y);
    array_uchar.push_back(it.z);
    array_uchar.push_back(it.w);
  }

  writeArray(name, array_uchar, dims_uchar4, compress);
}

// variable length strings are a special case and need specialization
template <>
inline void HDF5File::writeArray(const std::string &name,
                                 const std::vector<std::string> &array,
                                 const std::vector<int> &dims, bool compress) {
  // check dataset existence (hdf5 does not allow changes)
  if (checkVariableExists(name))
    throw std::runtime_error(std::string("HDF5File::writeArray: variable " +
                                         name + " already exists"));

  // check consistency of dims
  std::size_t n_elements = 1;
  for (const auto &dim : dims) {
    n_elements *= dim;
  }

  if (array.size() != n_elements)
    throw std::runtime_error(
        std::string("HDF5File::writeArray: inconsistent dims"));

  // convert string to const char*
  std::vector<const char *> c_array;
  for (auto &it : array)
    c_array.push_back(it.c_str());

  // create dataspace
  std::vector<hsize_t> dims_hsize;
  for (const auto &dim : dims)
    dims_hsize.push_back(dim);
  H5::DataSpace d_space(dims_hsize.size(), dims_hsize.data());

  // Variable length string
  H5::StrType datatype(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSet str_dataset = file_->createDataSet(name, datatype, d_space);

  str_dataset.write(c_array.data(), datatype);
}

// variable length strings are a special case and need specialization
template <>
inline void HDF5File::readArray(const std::string &name,
                                std::vector<std::string> &array,
                                std::vector<int> &dims) {
  try {
    H5::DataSet d_set(file_->openDataSet(name));
    H5::DataSpace d_space = d_set.getSpace();
    std::vector<hsize_t> dims_hsize(d_space.getSimpleExtentNdims());
    d_space.getSimpleExtentDims(dims_hsize.data());
    dims.clear();
    std::size_t n_elements = 1;
    for (auto &dim : dims_hsize) {
      dims.push_back(dim);
      n_elements *= dim;
    }
    if (array.size() != n_elements)
      array.resize(n_elements);
    //    if (n_elements == 1)
    //    {
    //      H5::StrType datatype(H5::PredType::C_S1, H5T_VARIABLE);
    //      char *buffer;
    //      d_set.read(&buffer, datatype, d_space);
    //      array.at(0) = buffer;
    //    } else
    //    {
    //      int buff_size = 500;
    //      H5::StrType datatype(H5::PredType::C_S1, buff_size);
    //      char buffer[n_elements][buff_size];
    //      d_set.read(buffer, datatype, d_space);
    //      for(int i=0;i<n_elements;i++)
    //        array.at(i) = buffer[i];
    //    }
    {
      int buff_size = 500;
      H5::StrType datatype(H5::PredType::C_S1, H5T_VARIABLE);
      char *buffer[n_elements];
      d_set.read((void *)buffer, datatype);
      for (int i = 0; i < n_elements; i++)
        array.at(i) = buffer[i];
    }
  }
  catch (H5::FileIException error) {
    error.printError();
    throw std::runtime_error(std::string("HDF5File::readArray: variable " +
                                         name + " doesn't seem to exist"));
  }
}
}

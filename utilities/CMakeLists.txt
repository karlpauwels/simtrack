cmake_minimum_required(VERSION 2.8.7)
project(utilities)

find_package(catkin REQUIRED COMPONENTS
)

find_package(HDF5 COMPONENTS CXX)
find_package(Boost COMPONENTS system filesystem REQUIRED)

find_package(CUDA 6.5 REQUIRED)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50)
set(GENCODE_SM52 -gencode=arch=compute_52,code=sm_52)
set(GENCODE_SM61 -gencode=arch=compute_61,code=sm_61)
set(CUDA_NVCC_FLAGS -O3;--compiler-options;-fPIC;-ftz=true;-prec-sqrt=false;-prec-div=false)
set(CUDA_PROPAGATE_HOST_FLAGS OFF) # don't propagate c++11 options to nvcc
if (${CUDA_VERSION_MAJOR} LESS 7)
    set(CUDA_OPTIONS ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${VERBOSE_PTXAS})
elseif(${CUDA_VERSION_MAJOR} LESS 8)
    set(CUDA_OPTIONS ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${GENCODE_SM50} ${GENCODE_SM52} ${VERBOSE_PTXAS})
else()
    set(CUDA_OPTIONS ${GENCODE_SM30} ${GENCODE_SM35} ${GENCODE_SM50} ${GENCODE_SM52} ${GENCODE_SM61} ${VERBOSE_PTXAS})
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

catkin_package(
  INCLUDE_DIRS include ${CUDA_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
)

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../README.md
)

include_directories(include
  ${HDF5_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ../third_party/nvidia_cub/
)

cuda_add_library(${PROJECT_NAME}
  src/cub_radix_sorter_kernels.cu
  src/utilities.cpp
  OPTIONS ${CUDA_OPTIONS}
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${Boost_LIBRARIES}
  ${CUDA_LIBRARIES}
)

catkin_add_gtest(test_utilities src/test/utest.cpp)
target_link_libraries(test_utilities
  ${PROJECT_NAME}
)

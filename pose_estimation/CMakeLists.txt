cmake_minimum_required(VERSION 2.8.7)
project(pose_estimation)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  siftgpu
  rendering
  utilities
)

find_package(CUDA 6.5 REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})

set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50)
set(GENCODE_SM52 -gencode=arch=compute_52,code=sm_52)
set(CUDA_NVCC_FLAGS -O3;--compiler-options;-fPIC;-ftz=true;-prec-sqrt=false;-prec-div=false)
set(CUDA_PROPAGATE_HOST_FLAGS OFF) # don't propagate c++11 options to nvcc
if (${CUDA_VERSION_MAJOR} LESS 7)
    set(CUDA_OPTIONS ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${VERBOSE_PTXAS})
else()
    set(CUDA_OPTIONS ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${GENCODE_SM50} ${GENCODE_SM52} ${VERBOSE_PTXAS})
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

catkin_package(
  INCLUDE_DIRS include ${CUDA_INCLUDE_DIRS}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS siftgpu rendering utilities
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
  ${catkin_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
)

cuda_add_library(${PROJECT_NAME}
  src/d_multiple_rigid_poses.cpp
  src/d_multiple_rigid_pose_sparse.cpp
  src/multiple_rigid_models_ogre.cpp
  src/multiple_rigid_pose_kernels.cu
  src/translation_rotation_3d.cpp
  src/normal_equations.cpp
  src/utility_kernels_pose.cu
  src/d_point_cloud.cpp
  OPTIONS ${CUDA_OPTIONS}
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
)

catkin_add_gtest(test_pose_estimation src/test/utest.cpp)
target_link_libraries(test_pose_estimation
  ${PROJECT_NAME}
)

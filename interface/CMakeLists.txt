cmake_minimum_required(VERSION 2.8.7)
project(interface)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  cv_bridge
  low_level_vision
  pose_estimation
)

find_package(HDF5 COMPONENTS CXX)
find_package(OpenCV 3.1.0 REQUIRED)
find_package(Eigen3 REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(CUDA 6.5 REQUIRED)
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
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS low_level_vision pose_estimation
  DEPENDS cv_bridge
)

# show additional files in qtcreator
execute_process(COMMAND find ${PROJECT_SOURCE_DIR} -type f
  OUTPUT_VARIABLE FILES_STRING)
string(REPLACE "\n" ";" FILES_LIST ${FILES_STRING})
add_custom_target(additional_files_${PROJECT_NAME} SOURCES
  ${FILES_LIST}
  ${PROJECT_SOURCE_DIR}/../README.md
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/multi_rigid_tracker.cpp
  src/multi_rigid_detector.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${HDF5_LIBRARIES}
  ${CUDA_LIBRARIES
)

add_executable(cmd_line_generate_sift_model
  src/cmd_line_generate_sift_model.cpp
)
target_link_libraries(cmd_line_generate_sift_model
  ${PROJECT_NAME}
)

add_executable(cmd_line_optical_flow
  src/cmd_line_optical_flow.cpp
)
target_link_libraries(cmd_line_optical_flow
  ${PROJECT_NAME}
)

add_executable(cmd_line_render
  src/cmd_line_render.cpp
)
target_link_libraries(cmd_line_render
  ${PROJECT_NAME}
)

add_executable(cmd_line_render_scene
  src/cmd_line_render_scene.cpp
)
target_link_libraries(cmd_line_render_scene
  ${PROJECT_NAME}
)

add_executable(cmd_line_get_robot_frame
  src/cmd_line_get_robot_frame.cpp
)
target_link_libraries(cmd_line_get_robot_frame
  ${PROJECT_NAME}
)

add_executable(cmd_line_extract_sift
  src/cmd_line_extract_sift.cpp
)
target_link_libraries(cmd_line_extract_sift
  ${PROJECT_NAME}
)

add_executable(cmd_line_detect_object_pose
  src/cmd_line_detect_object_pose.cpp
)
target_link_libraries(cmd_line_detect_object_pose
  ${PROJECT_NAME}
)

add_executable(cmd_line_track_multi_object
  src/cmd_line_track_multi_object.cpp
)
target_link_libraries(cmd_line_track_multi_object
  ${PROJECT_NAME}
)

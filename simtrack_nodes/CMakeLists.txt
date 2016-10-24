cmake_minimum_required(VERSION 2.8.7)
project(simtrack_nodes)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  cv_bridge
  image_transport
  roscpp
  dynamic_reconfigure
  std_msgs
  message_generation
  tf
  geometry_msgs
  interface
)

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

generate_dynamic_reconfigure_options(
  cfg/Visualization.cfg
)

add_service_files(
  FILES
  SwitchCamera.srv
  SwitchObjects.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  DEPENDS cv_bridge
  CATKIN_DEPENDS image_transport tf geometry_msgs message_runtime interface
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
  ${CUDA_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/pr2_cam_switch_node.cpp
  src/multi_rigid_node.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${CUDA_LIBRARIES}
)

add_executable(multi_rigid_node
  src/multi_rigid_node_main.cpp
)
target_link_libraries(multi_rigid_node
  ${PROJECT_NAME}
)
add_dependencies(multi_rigid_node
  ${PROJECT_NAME}_gencfg)

add_executable(pr2_cam_switch_node
  src/pr2_cam_switch_node_main.cpp
)
target_link_libraries(pr2_cam_switch_node
  ${PROJECT_NAME}
)

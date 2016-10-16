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
  ${OpenCV2_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
)

add_library(${PROJECT_NAME}
  src/pr2_cam_switch_node.cpp
  src/multi_rigid_node.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
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

cmake_minimum_required(VERSION 2.8.7)
project(rendering)

find_package(catkin REQUIRED COMPONENTS
  cmake_modules
  resource_retriever
  kdl_parser
)

find_package(CUDA 6.5 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV 2.4.8 REQUIRED)
find_package(Eigen REQUIRED)
add_definitions(${EIGEN_DEFINITIONS})
find_package(urdfdom REQUIRED)
find_package(orocos_kdl REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLEW REQUIRED glew)

pkg_check_modules(OGRE_OV OGRE OGRE-Overlay)

# Old versions of OGRE (pre 1.9) included OGRE-Overlay in the main package
# (i.e. there was no OGRE-Overlay component).  So if the above
# pkg_check_modules() failed, try looking for just OGRE.
if(NOT OGRE_OV_FOUND)
  pkg_check_modules(OGRE_OV REQUIRED OGRE)
endif(NOT OGRE_OV_FOUND)

## Find OGRE Plugin path (not necessarily platform-independent, I guess)
execute_process(COMMAND
  pkg-config --variable=plugindir OGRE
  OUTPUT_VARIABLE OGRE_PLUGIN_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )
#message(STATUS "OGRE_PLUGIN_PATH=${OGRE_PLUGIN_PATH}")

# Build the version number and other build-time constants into the
# source for access at run-time.
set(ENV_CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/env_config.cpp)
#set(RVIZ_VERSION ${rviz_VERSION})
#message("making version ${RVIZ_VERSION}.")
set(ROS_DISTRO $ENV{ROS_DISTRO})

configure_file(src/env_config.cpp.in ${ENV_CONFIG_FILE} @ONLY)

# find absolute path of ogre libraries.
# This is stored in the cache to save time on cmake re-run.
# Using absolute paths is necessary if pkg-config finds Ogre in a different
# location than the default.  This can happen when Ogre is built from source,
# or when 2 versions of Ogre are installed.  Using absolute paths ensures that
# components that link against Ogre (rviz and any packages depending on rviz)
# all find the same Ogre shared library.
if(NOT DEFINED OGRE_OV_LIBRARIES_ABS)
  unset(OGRE_OV_LIBRARIES_ABS_TMP)
  foreach(_lib ${OGRE_OV_LIBRARIES})
    set(OGRE_OV_LIB_TAG "OGRE_OV_RVIZ_LIB_${_lib}")

    find_library(${OGRE_OV_LIB_TAG}
            NAMES ${_lib}
            HINTS ${OGRE_OV_LIBRARY_DIRS}
            PATHS ${OGRE_OV_LIBRARY_DIRS}
    )

    set(OGRE_OV_LIBRARIES_ABS_TMP ${OGRE_OV_LIBRARIES_ABS_TMP} ${${OGRE_OV_LIB_TAG}})
  endforeach(_lib)

  set(OGRE_OV_LIBRARIES_ABS ${OGRE_OV_LIBRARIES_ABS_TMP}
        CACHE FILEPATH "Pathname of library ${_lib}")
endif(NOT DEFINED OGRE_OV_LIBRARIES_ABS)
# add gl rendersystem
list(APPEND OGRE_OV_LIBRARIES_ABS
  ${OGRE_PLUGIN_PATH}/RenderSystem_GL.so
)
message(STATUS "OGRE_OV_LIBRARIES_ABS=${OGRE_OV_LIBRARIES_ABS}")

find_package(ASSIMP QUIET)
if (NOT ASSIMP_FOUND)
  pkg_check_modules(ASSIMP assimp)
endif()
if (ASSIMP_FOUND)
  if( NOT ${ASSIMP_VERSION} VERSION_LESS "2.0.1150" )
    add_definitions(-DASSIMP_UNIFIED_HEADER_NAMES)
    message(STATUS "Assimp version has unified headers")
  else()
    message(STATUS "Assimp version does not have unified headers")
  endif()
  include_directories(${ASSIMP_INCLUDE_DIRS})
  link_directories(${ASSIMP_LIBRARY_DIRS})
else()
  message(STATUS "could not find assimp (perhaps available through ROS package?), so assimping assimp v2")
  set(ASSIMP_LIBRARIES assimp)
  set(ASSIMP_LIBRARY_DIRS)
  set(ASSIMP_CXX_FLAGS)
  set(ASSIMP_CFLAGS_OTHER)
  set(ASSIMP_LINK_FLAGS)
  set(ASSIMP_INCLUDE_DIRS)
endif()


SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

catkin_package(
  INCLUDE_DIRS
    include
    ${OGRE_OV_INCLUDE_DIRS}
  LIBRARIES
    ${PROJECT_NAME}
    ${OGRE_OV_LIBRARIES_ABS}
    ${urdfdom_LIBRARIES}
  DEPENDS libx11-dev opengl libglew-dev cv_bridge orocos_kdl
  CATKIN_DEPENDS
    resource_retriever
    kdl_parser
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
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV2_INCLUDE_DIRS}
  ${EIGEN_INCLUDE_DIR}
  ${urdfdom_INCLUDE_DIRS}
  ${OGRE_OV_INCLUDE_DIRS}
  ${orocos_kdl_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/windowless_gl_context.cpp
  src/robot_link.cpp
  src/rigid_object.cpp
  src/ogre_context.cpp
  src/ogre_multi_render_target.cpp
  src/stl_loader.cpp
  src/mesh_loader.cpp
  src/robot.cpp
  ${ENV_CONFIG_FILE}
)

target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${GLEW_LIBRARIES}
  ${OPENGL_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${urdfdom_LIBRARIES}
  ${OGRE_OV_LIBRARIES_ABS}
  ${orocos_kdl_LIBRARY_DIRS}
  assimp
)

catkin_add_gtest(test_rendering src/test/utest.cpp)
target_link_libraries(test_rendering
  ${PROJECT_NAME}
)

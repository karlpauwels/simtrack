cmake_minimum_required(VERSION 2.8.7)
project(siftgpu)

find_package(catkin REQUIRED COMPONENTS
)

find_package(PkgConfig)
pkg_check_modules(GLEW REQUIRED glew)
find_package(CUDA 6.5 REQUIRED)

set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35)
set(GENCODE_SM50 -gencode=arch=compute_50,code=sm_50)
set(GENCODE_SM52 -gencode=arch=compute_52,code=sm_52)
set(CUDA_NVCC_FLAGS -O3;-ftz=true;-prec-sqrt=false;-prec-div=false)
set(CUDA_PROPAGATE_HOST_FLAGS ON) # propagate -DCUDA_SIFTGPU_ENABLED
if (${CUDA_VERSION_MAJOR} LESS 7)
    set(CUDA_OPTIONS ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${VERBOSE_PTXAS})
else()
    set(CUDA_OPTIONS ${GENCODE_SM20} ${GENCODE_SM30} ${GENCODE_SM35} ${GENCODE_SM50} ${GENCODE_SM52} ${VERBOSE_PTXAS})
endif()

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUDA_SIFTGPU_ENABLED")
# disable warnings thrown by siftgpu compilation
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-int-to-pointer-cast -Wno-write-strings -Wno-unused-result")

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  DEPENDS libglew-dev
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
)

cuda_add_library(${PROJECT_NAME}
  src/ProgramCU.cu
  src/CuTexImage.cpp
  src/FrameBufferObject.cpp
  src/GlobalUtil.cpp
  src/GLTexImage.cpp
  src/ProgramGLSL.cpp
  src/ProgramGPU.cpp
  src/PyramidCU.cpp
  src/PyramidGL.cpp
  src/ShaderMan.cpp
  src/SiftGPU.cpp
  src/SiftMatch.cpp
  src/SiftMatchCU.cpp
  src/SiftPyramid.cpp
  OPTIONS ${CUDA_OPTIONS}
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${GLEW_LIBRARIES}
)

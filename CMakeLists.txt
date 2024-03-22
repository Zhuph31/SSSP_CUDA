cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")


project(sssp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# thrust 
set(Thrust_DIR /usr/include/thrust/cmake)
set(CUB_DIR /usr/include/cub/cmake)
include_directories(${Thrust_DIR} ${CUB_DIR})
find_package(Thrust REQUIRED CONFIG) 
thrust_create_target(Thrust)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

option(USE_CUDA "Use CUDA" ON)

file(GLOB cpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cc")
file(GLOB gpu_source_files "${CMAKE_SOURCE_DIR}/src/*.cu")

# Modern CMake
if(USE_CUDA)
    enable_language("CUDA")
    add_executable(sssp ${cpu_source_files} ${gpu_source_files})
else(USE_CUDA)
    add_executable(sssp ${cpu_source_files})
endif()

message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()


target_link_libraries(sssp Thrust)
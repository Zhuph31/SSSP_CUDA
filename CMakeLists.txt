cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(sssp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
cmake_minimum_required(VERSION 3.10)

cmake_minimum_required(VERSION 3.10)

#set(CMAKE_C_STANDARD 11)
#set(CMAKE_C_STANDARD_REQUIRED True)

project(vector)
add_executable(vector vector.c)
include_directories("/opt/rocm/hip/include")
list(APPEND CMAKE_PREFIX_PATH)

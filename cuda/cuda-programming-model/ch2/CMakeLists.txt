cmake_minimum_required (VERSION 2.6)
project (p33 LANGUAGES C CUDA)

# The version number.

set (test_VERSION_MAJOR 1)
set (test_VERSION_MINOR 0)

option(DYN_BUILD "Dynamic build" ON)
find_package(CUDA QUIET REQUIRED)

set(SOURCES
    P33.cu
    lib1/lib1.c
)

add_library(lib1 SHARED lib1/lib1.c)
#target_include_directories(p33 PUBLIC lib1/)

add_executable(${SOURCES})
#target_link_libraries(p33 PRIVATE lib1)


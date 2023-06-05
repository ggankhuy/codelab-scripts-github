cmake_minimum_required (VERSION 2.6)
project (p53 LANGUAGES C CUDA)

#include(CheckLanguage)
#check_language(CUDA)

# The version number.

set (test_VERSION_MAJOR 1)
set (test_VERSION_MINOR 0)

option(DYN_BUILD "Dynamic build" ON)
find_package(CUDA QUIET REQUIRED)

#set(SOURCES
#    P53.cu
#)

#add_library(lib1 SHARED lib1/lib1.c)
#target_include_directories(p53 PUBLIC lib1/)

add_executable(P53 p53.cu)
#target_link_libraries(p53 lib1)


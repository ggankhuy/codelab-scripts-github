cmake_minimum_required (VERSION 2.6)
project (p35)

# The version number.

set (test_VERSION_MAJOR 1)
set (test_VERSION_MINOR 0)

# should we use our own math functions?

option(DYN_BUILD "Dynamic build" OFF)

find_package(CUDA QUIET REQUIRED)

set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )

SET(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
SET(CMAKE_CXX_COMPILER CMAKE_CUDA_COMPILER)

#cuda_add_executable(
#    P35
#    P35.cu
#)

add_executable(p35 p35.cu)
set_target_properties(p35 PROPERTIES LINKER_LANGUAGE CXX)


# configure a header file to pass some of the CMake settings
# to the source code

#configure_file (
#  "${PROJECT_SOURCE_DIR}/testConfig.h.in"
#  "${PROJECT_BINARY_DIR}/testConfig.h"
#  )
 
# add the binary tree to the search path for include files
# so that we will find testConfig.h

#include_directories("${PROJECT_BINARY_DIR}")
 
# add the MathFunctions library?
#
#if (USE_MYMATH)
#  include_directories ("${PROJECT_SOURCE_DIR}/MathFunctions")
#  add_subdirectory (MathFunctions)
#  set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
#endif (USE_MYMATH)

#add the executable

#target_link_libraries(test PUBLIC all)

#if(DYN_BUILD)
#  message("Building shared library...")
#  add_library(all SHARED libfile1.c libfile2.c libfile3.c)
#else (DYN_BUILD)
#  message("Building static library...")
#  add_library(lib1 STATIC libfile1.c libfile2.c libfile3.c)
#endif (DYN_BUILD)

#add_custom_command(OUTPUT output1 COMMAND nvcc *.o -shared -o liball.so)

#install (TARGETS MathFunctions DESTINATION bin)
#install (FILES MathFunctions.h DESTINATION include)

#include(CTest)

# does the application run
#add_test (testRuns test 25)
# does it sqrt of 25
#add_test (testComp25 test 25)
##set_tests_properties (testComp25 PROPERTIES PASS_REGULAR_EXPRESSION "25 is 5")
# does it handle negative numbers
#add_test (testNegative test -25)
#set_tests_properties (testNegative PROPERTIES PASS_REGULAR_EXPRESSION "-25 is 0")
# does it handle small numbers
#add_test (testSmall test 0.0001)
#set_tests_properties (testSmall PROPERTIES PASS_REGULAR_EXPRESSION "0.0001 is 0.01")
# does the usage message work?
#add_test (testUsage test)
#set_tests_properties (testUsage PROPERTIES PASS_REGULAR_EXPRESSION "Usage:.*number")

#define a macro to simplify adding tests, then use it
#macro (do_test arg result)
  #add_test (testComp${arg} test ${arg})
  #set_tests_properties (testComp${arg}
#    PROPERTIES PASS_REGULAR_EXPRESSION ${result})
#endmacro (do_test)
 
# do a bunch of result based tests
#do_test (25 "25 is 5")
#do_test (-25 "-25 is 0")

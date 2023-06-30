cmake_minimum_required (VERSION 2.6)
set(MAINFILE tensile_dat_reader)
set(TENSILE_ROOT /root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile)
set(TENSILE_SRC_ROOT $TENSILE_ROOT/Tensile/Source/lib/source)
set(CXX_FLAGS -std=c++17 -I$TENSILE_ROOT/Tensile/Source/lib/include/  -c)
project (${MAINFILE})

# The version number.

set (test_VERSION_MAJOR 1)
set (test_VERSION_MINOR 0)

# should we use our own math functions?

option(DYN_BUILD "Dynamic build" ON)


include_directories("${PROJECT_BINARY_DIR}")
 
add_executable(${MAINFILE} ${MAINFILE}.c)
target_link_libraries(q1 PUBLIC all)

set (LIBFILES \
        ${TENSILE_SRC_ROOT}/Tensile.cpp \
        ${TENSILE_SRC_ROOT}/Debug.cpp \
        ${TENSILE_SRC_ROOT}/msgpack/MessagePack.cpp \
        ${TENSILE_SRC_ROOT}/ContractionSolution.cpp \
        ${TENSILE_SRC_ROOT}/ContractionProblem.cpp \
        ${TENSILE_SRC_ROOT}/TensorOps.cpp \
        ${TENSILE_SRC_ROOT}/DataTypes.cpp \
        ${TENSILE_SRC_ROOT}/AMDGPU.cpp  \
        ${TENSILE_SRC_ROOT}/Utils.cpp  \
        ${TENSILE_SRC_ROOT}/KernelArguments.cpp \
        ${TENSILE_SRC_ROOT}/ArithmeticUnitTypes.cpp \
        ${TENSILE_SRC_ROOT}/EmbeddedData.cpp \
        ${TENSILE_SRC_ROOT}/EmbeddedLibrary.cpp \
        ${TENSILE_SRC_ROOT}/KernelLanguageTypes.cpp \
        ${TENSILE_SRC_ROOT}/PerformanceMetricTypes.cpp \
        ${TENSILE_SRC_ROOT}/ScalarValueTypes.cpp \
        ${TENSILE_SRC_ROOT}/TensorDescriptor.cpp
)
if(DYN_BUILD)
  message("Building shared library...")
  add_library(all SHARED ${LIBFILES})
else (DYN_BUILD)
  message("Building static library...")
  add_library(lib1 STATIC ${LIBFILES})
endif (DYN_BUILD)

add_custom_command(OUTPUT output1 COMMAND hipcc *.o -shared -o liball.so)

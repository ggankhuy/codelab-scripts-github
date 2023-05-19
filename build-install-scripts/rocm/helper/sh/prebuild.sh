# import function definitions.
echo "prebuild.sh entered..."
source sh/common.sh

# setup all environment related variables.

NPROC=`nproc`
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
ROCM_INST_FOLDER=/opt/rocm/

# setup all version related variables.

VERSION_MAJOR=$vermajor
VERSION_MINOR=$verminor
VERSION=$VERSION_MAJOR.$VERSION_MINOR
echo "major/minor version: $VERSION_MAJOR/$VERSION_MINOR"

# setup log paths.

LOG_DIR=/log/rocmbuild/
LOG_SUMMARY=$LOG_DIR/build-summary.log
LOG_SUMMARY_L2=$LOG_DIR/build-summary-L2.log
mkdir $LOG_DIR -p

# setup all paths related variables except log.

ROCM_INST_FOLDER=/opt/rocm/
ROCM_SRC_FOLDER=/root/gg/git/ROCm-$VERSION_MAJOR/
export ROCM_SRC_FOLDER=$ROCM_SRC_FOLDER
echo "ROCM_SRC_FOLDER: $ROCM_SRC_FOLDER"

# setup all build related configurations, switches etc.,

CONFIG_TEST=0
FAST_INSTALL=0
ESSENTIAL_INSTALL=0
CONFIG_BUILD_PACKAGE=0
CONFIG_BYPASS_LLVM=0
CONFIG_DISABLE_rocSOLVER=1
CONFIG_DISABLE_hipBLAS=1
CONFIG_MIOPEN_BUILD_ROCBLAS=" -DMIOPEN_USE_ROCBLAS=off" 
CONFIG_TENSILE_INSTALL_PIP=0

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [[ -z $VERSION ]] ; then echo "You need to specify at least major version" ; exit 1 ; fi

CONFIG_BUILD_LLVM=1
CONFIG_BUILD_PY=0
CONFIG_BUILD_CMAKE=0
CONFIG_BUILD_PACKAGE=0
CONFIG_BUILD_FAST=0
CONFIG_TEST_MODE=0
DEBUG=1

if [[ $CONFIG_TEST_MODE -eq 1 ]]; then
    echo "TEST_MODE: sh/build.sh is called with parameters: '$@'"
    exit 0
fi
if [[ $DEBUG -eq 1 ]] ; then
    echo "DBG: sh/build.sh is called with parameters: '$@'"
fi

if [[ $CONFIG_BUILD_PACKAGE -ne 0 ]] ; then
    echo "will build packages..."
    CONFIG_BUILD_PKGS_LOC=/rocm-packages/
    BUILD_TARGET=package
    INSTALL_SH_PACKAGE="-p"
    INSTALL_TARGET=package
    mkdir -p $CONFIG_BUILD_PKGS_LOC
else
    echo "will not build packages..."
    BUILD_TARGET=""
    INSTALL_SH_PACKAGE=""
    INSTALL_TARGET=install
fi

# FAST build options.

FAST_BUILD_ROCBLAS_OPT=" -icd "

if [[ $CONFIG_BUILD_FAST -eq 1 ]] ; then
    FAST_BUILD_ROCBLAS_OPT=" -ida gf908 -l asm_full "
fi

# Componentwise build options. Some of them are defined to get build working.

CONFIG_MIOPEN_BUILD_ROCBLAS=" -DMIOPEN_USE_ROCBLAS=off" 
CONFIG_TENSILE_INSTALL_PIP=0

# import function definitions.
echo "prebuild.sh entered..."
source sh/common.sh

# setup all environment related variables.

NPROC=`nproc`
PWD=`pwd`
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

# setup all version related variables.

VERSION_MAJOR=$vermajor
VERSION_MINOR=$verminor
VERSION=$VERSION_MAJOR.$VERSION_MINOR
echo "major/minor version: $VERSION_MAJOR/$VERSION_MINOR"

# error codes.

ERROR_CODE_OK=0
ERROR_CODE_ROCM_SRC=100
ERROR_CODE_ROCM_SRC_SCRIPT=101
ERROR_CODE_VERSION=102

# setup log paths.

LOG_DIR=/log/rocmbuild/
LOG_SUMMARY=$LOG_DIR/build-summary.log
LOG_SUMMARY_L2=$LOG_DIR/build-summary-L2.log
mkdir $LOG_DIR -p

# setup src folder and its variables.

ROCM_SRC_SCRIPT=rocm-source.sh
ROCM_SRC_FOLDER=$PWD/ROCm-$VERSION_MAJOR/
echo ROCM_SRC_FOLDER: $ROCM_SRC_FOLDER

if [[ ! -f $ROCM_SRC_FOLDER ]] ; then
        echo "$ROCM_SRC_FOLDER does not exist."
    if [[ ! -f $PWD/sh/$ROCM_SRC_SCRIPT ]] ; then
        echo "$PWD/sh/$ROCM_SRC_SCRIPT???"
        echo "I did not see ROCm source checked out in current directory nor I see rocm-source.sh. Can not continue."
        exit $ERROR_CODE_ROCM_SRC
    else
        echo "Checkout ROCm source..."
        ./sh/$ROCM_SRC_SCRIPT $VERSION_MAJOR
    fi
else
    echo "I do see $ROCM_SRC_FOLDER present. Assuming all ROCm source files are checked out successfully. Continuing..."
fi

export ROCM_SRC_FOLDER=$ROCM_SRC_FOLDER
echo "ROCM_SRC_FOLDER: $ROCM_SRC_FOLDER"

# setup all paths related variables except log.

ROCM_INST_FOLDER=/opt/rocm/

# setup all build related options.
# do not put here any command line switches coming from python code.

CONFIG_DISABLE_rocSOLVER=1
CONFIG_DISABLE_hipBLAS=1
CONFIG_MIOPEN_BUILD_ROCBLAS=" -DMIOPEN_USE_ROCBLAS=off" 
CONFIG_TENSILE_INSTALL_PIP=0

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

if [[ -z $VERSION ]] ; then echo "You need to specify at least major version" ; exit ERROR_CODE_VERSION ; fi

DEBUG=1

if [[ $CONFIG_TEST_MODE -eq 1 ]]; then
    echo "TEST_MODE: sh/build.sh is called with parameters: '$@'"
    exit ERROR_CODE_OK
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

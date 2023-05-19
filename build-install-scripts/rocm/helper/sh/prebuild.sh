# import function definitions.

source sh/common.sh

# setup all environment related variables.

CONFIG_TEST=0
FAST_INSTALL=0
ESSENTIAL_INSTALL=0
CONFIG_BUILD_PACKAGE=0
CONFIG_BYPASS_LLVM=0
CONFIG_DISABLE_rocSOLVER=1
CONFIG_DISABLE_hipBLAS=1

LOG_DIR=./log/rocmbuild/
NPROC=`nproc`

ROCM_INST_FOLDER=/opt/rocm/

# attempt to determine rocm version.
# if rocm is already installed, it will set the version on installed version.
# if rocm is not already installed, exit with error.
# At this point, we havent tested if build will be successful without rocm installation, nor test.

VERSION_STRING=`cat /opt/rocm/.info/version`
if [[ -z $VERSION_STRING ]] ; then echo "I can not determine ROCm version. Install ROCm first." ; exit 1 ; fi 

ROCM_VERSION=`echo $VERSION_STRING | cut -d '-' -f1`
echo "VERSION_STRING: $VERSION_STRNG"
MAJOR_VERSION=

ROCM_SRC_FOLDER=/root/gg/git/ROCm-$VERSION/
echo "ROCM_SRC_FOLDER: $ROCM_SRC_FOLDER, minor version: $MINOR_VERSION"
export ROCM_SRC_FOLDER=$ROCM_SRC_FOLDER

# these are settings both common to shell and python.

LOG_SUMMARY=$LOG_DIR/build-summary.log
LOG_SUMMARY_L2=$LOG_DIR/build-summary-l2.log

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

VERSION=$vermajor
MINOR_VERSION=$verminor
mkdir $LOG_DIR -p

if [[ -z $VERSION ]] ; then echo "You need to specify at least major version" ; exit 1 ; fi


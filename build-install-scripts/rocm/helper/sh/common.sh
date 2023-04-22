CONFIG_TEST=0
FAST_INSTALL=0
ESSENTIAL_INSTALL=0
CONFIG_BUILD_PACKAGE=0
CONFIG_BYPASS_LLVM=0
CONFIG_DISABLE_rocSOLVER=1
CONFIG_DISABLE_hipBLAS=1

function install_pip_libs() {
    for i in $@; do
        echo =======================
        pip3 install $i
    done
    
}
function install_python() {
    #Python defs.

    CURR_VER=`python3 --version  | cut -d ' ' -f2`
    PYTHON_VER_MAJOR=3.9
    PYTHON_VER_MINOR=10

    PYTHON_VER=$PYTHON_VER_MAJOR.$PYTHON_VER_MINOR
    PYTHON_FULL_NAME=Python-$PYTHON_VER
    PYTHON_TAR=$PYTHON_FULL_NAME.tgz

    if [[ $CURR_VER == $PYTHON_VER ]] ; then
        echo "Current installed version is same as the one being installed..., exiting"
        return 0
    else
        echo "Installing..."
    fi

    sudo yum -y install epel-release
    sudo yum update -y
    sudo yum groupinstall "Development Tools" -y

    sudo yum install openssl-devel libffi-devel bzip2-devel -y
    wget -nc https://www.python.org/ftp/python/$PYTHON_VER/$PYTHON_TAR
    tar -xvf $PYTHON_TAR
    cd $PYTHON_FULL_NAME

    if [[ $? -ne 0 ]] ; then
        echo "Can not cd into $PYTHON_VER directory..."
        exit 1
    fi
    ./configure --enable-optimizations
    sudo make -j`nproc` install

    echo "Testing the installation..."
    python$PYTHON_VER_MAJOR --version
    if [[ $? -ne 0 ]] ; then
        echo "Unable to find 3.9"
    fi
    PATH_PYTHON_U=`which python$PYTHON_VER_MAJOR`
    echo "new path: $PATH_PYTHON_U"
    rm -rf /usr/bin/python
    echo ln -s $PATH_PYTHON_U /usr/bin/python
    ln -s $PATH_PYTHON_U /usr/bin/python

    rm -rf /usr/bin/python3
    ln -s /usr/bin/python /usr/bin/python3
    cd ..
}

function SINGLE_LINE() {
    local counter
    for counter in 40 ; do echo -ne "-" 2>&1 | tee -a $LOG_SUMMARY ; done
}
 
function DOUBLE_LINE() {
    local counter
    for counter in 40 ; do echo -ne "=" 2>&1 | tee -a $LOG_SUMMARY ; done
} 

function build_entry () {
    t1=$SECONDS
    L_CURR_BUILD=$1
    DOUBLE_LINE
    echo " Building entry: $L_CURR_BUILD" 2>&1 | tee -a $LOG_SUMMARY
    DOUBLE_LINE
}

function build_exit() {
    t2=$SECONDS
    echo Build took $((t2-t1)) seconds 2>&1 | tee -a $LOG_SUMMARY
}

function setup_root_rocm_softlink () {
    rm ~/ROCm
    ln -s $ROCM_SRC_FOLDER  ~/ROCm
    if [[ $? -ne  0 ]] ; then 
        echo "Error during setting up the softlink ~/ROCm"
        ls -l ~
        exit 1
    fi
}

LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
#ROCM_SRC_FOLDER=~/ROCm-$VERSION
ROCM_INST_FOLDER=/opt/rocm-$VERSION.$MINOR_VERSION

# these are settings both common to shell and python.

LOG_SUMMARY=$LOG_DIR/build-summary.log
LOG_SUMMARY_L2=$LOG_DIR/build-summary-l2.log

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

VERSION="5.2"
MINOR_VERSION="0"
VERSION=$vermajor
MINOR_VERSION=$verminor
mkdir /log/rocmbuild/ -p
ROCM_SRC_FOLDER=/root/gg/git/ROCm-$VERSION/
echo "ROCM_SRC_FOLDER: $ROCM_SRC_FOLDER, minor version: $MINOR_VERSION"
export ROCM_SRC_FOLDER=$ROCM_SRC_FOLDER


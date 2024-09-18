
set -x

CONFIG_DO_PREREQ=0
CONFIG_BUILD_ROCT=1
CONFIG_BUILD_KFDTEST=1
CMAKE_OPTS="--trace" 
CMAKE_OPTS="" 
LOG_DIR_SUFFIX="ROCT+KFDTEST"
ROCM_VERSION=6.2
ROCM_VERSION_INSTALLED=6.2
if [[ ! -d ROCT-Thunk-Interface ]] ; then
    git clone https://github.com/ROCm/ROCT-Thunk-Interface.git
else
    echo "bypassing git clone as it exists already..."
fi

LOG_DIR=`pwd`/log/$CMAKE_OPTS_$LOG_DIR_SUFFIX
mkdir $LOG_DIR -p
pushd ROCT-Thunk-Interface
git checkout rocm-$ROCM_VERSION.x

if [[ $CONFIG_DO_PREREQ == 1 ]] ; then
    for i in ghc-terminfo-devel libzstd-devel.x86_64 rocm-llvm-devel rpm-build dpkg libdrm-devel numactl-devel
    do
        sudo yum install $i -y
    done
fi
rm -rf log/*

if [[ $CONFIG_BUILD_ROCT == 1 ]] ; then
    rm -rf build
    mkdir build
    mkdir log
    pushd build
    cmake $CMAKE_OPTS  ..  2>&1 | tee $LOG_DIR/1.roct.cmake.log
    make -j16 package  2>&1 | tee $LOG_DIR/2.roct.make.log
    popd
    yum install --allowerasing --nobest `find . -name *rpm | head -1` -y
fi

if [[ $CONFIG_BUILD_KFDTEST == 1 ]] ; then
    pushd tests/kfdtest
    rm -rf build
    mkdir build
    cd build

    #export LIBHSAKMT_PATH=/opt/rocm/lib64/libhsakmt.so
    CMAKE_PREFIX_PATH=/opt/rocm/lib/llvm/lib/cmake/llvm/ cmake $CMAKE_OPTS .. 2>&1  | tee $LOG_DIR/3.kfdtest.cmake.log &&
    make -j16  2>&1  | tee $LOG_DIR/4.kfdtest.make.log
    popd
fi
popd


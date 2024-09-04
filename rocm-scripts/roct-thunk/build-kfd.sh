
set -x
git clone https://github.com/ROCm/ROCT-Thunk-Interface.git
cd ROCT-Thunk-Interface
git checkout rocm-6.1.x

LOG_DIR=`pwd`/log

for i in ghc-terminfo-devel libzstd-devel.x86_64 rocm-llvm-devel rpm-build dpkg libdrm-devel numactl-devel
do
    sudo yum install $i -y
done

rm -rf log/*
rm -rf build
mkdir build
mkdir log
pushd build
cmake ..  2>&1 | tee $LOG_DIR/1.roct.cmake.log
#make -j16  ..  2>&1 | tee $LOG_DIR/2.roct.make.log
make -j16 package  2>&1 | tee $LOG_DIR/2.roct.make.log
popd
yum install --allowerasing --nobest `find . -name *rpm | head -1` -y

pushd tests/kfdtest
rm -rf build
mkdir build
cd build

#export LIBHSAKMT_PATH=/opt/rocm/lib64/libhsakmt.so
CMAKE_PREFIX_PATH=/opt/rocm-6.1.0/lib/llvm/lib/cmake/llvm/ cmake .. 2>&1  | tee $LOG_DIR/3.kfdtest.cmake.log &&
#LD_LIBRARY_PATH=/opt/rocm/lib64/libhsakmt.so 
make -j16  2>&1  | tee $LOG_DIR/4.kfdtest.make.log
popd


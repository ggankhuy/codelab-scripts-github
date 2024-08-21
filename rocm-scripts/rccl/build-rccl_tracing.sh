set -x
mkdir log
./install-ninja.sh
DATE=`date +%Y%m%d-%H-%M-%S`
LOG_DIR=`pwd`/log/$DATE/
mkdir -p $LOG_DIR
rm -rf build
mkdir build
pushd build || exit 1

t1=$SECONDS
CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ -GNinja .. | tee $LOG_DIR/rccl.1.make.log
ninja -j32 2>&1 | tee $LOG_DIR/rccl.2.make.log
t2=$SECONDS
echo build time: $((t2-t1)) | tee $LOG_DIR/rccl.3.buildtime.log
popd
ninjatracing `find . -name .ninja_log` | tee $LOG_DIR/rccl.4.ninjatracing.json

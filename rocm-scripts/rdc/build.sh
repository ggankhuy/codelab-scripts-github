# this script just combines build-rdc.sh and build-grpc.sh in this same folder.
# those 2 can be removed once this one known to work.

OPTION_CLEAN_BUILD_GRPC=0
OPTION_CLEAN_BUILD_RDC=1

set -x 
#centos : gcrp part working.
#ubuntu: in test.

apt install -y automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl
apt install -y libyaml-cpp-dev libabsl-dev doxygen libcap-dev
apt install rocm-validation-suite -y

export GRPC_ROOT=/opt/grpc
export GRPC_PROTOC_ROOT=/opt/grpc

if [[ ! -d grpc ]] ; then 
    git clone -b v1.61.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
fi

pushd  grpc

if [[ $OPTION_CLEAN_BUILD_GRPC -eq 1 ]] ; then rm -rf build ; fi

cmake -B build \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_INSTALL_PREFIX=/opt/rocm/rdc \
    -DCMAKE_BUILD_TYPE=Release

make -C build -j $(nproc)
sudo make -C build install
echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf
popd

if [[ ! -z rdc ]] ; then
    git clone https://github.com/ROCm/rdc
fi
pushd rdc
pwd
mkdir -p build
cd build

if [[ $OPTION_CLEAN_BUILD_RDC -eq 1 ]] ; then rm -rf build ; fi

CMAKE_PREFIX_PATH=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/grpc/build/third_party/protobuf/cmake/protobuf/ \
cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT" -DBUILD_RVS=ON -DCMAKE_PREFIX_PATH=/opt/rocm-6.1.0-13435/lib/cmake/rvs/ ..

# other build cmake generator param options:
#-DGRPC_ROOT="$GRPC_PROTOC_ROOT" ..
#-DBUILD_STANDALONE=off 

make -j$(nproc) install

RDC_LIB_DIR=/opt/rocm/rdc/lib
GRPC_LIB_DIR=/opt/grpc/lib
echo -e "${GRPC_LIB_DIR}\n${GRPC_LIB_DIR}64" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
echo -e "${RDC_LIB_DIR}\n${RDC_LIB_DIR}64" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
ldconfig

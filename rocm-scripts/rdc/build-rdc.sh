apt install -y automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl
apt install -y libyaml-cpp-dev libabsl-dev
set -x 
if [[ ! -z rdc ]] ; then
    git clone https://github.com/ROCm/rdc
fi
pushd rdc
pwd
mkdir -p build
cd build
rm -rf ./*

GRPC_PROTOC_ROOT=/opt/grpc

CMAKE_PREFIX_PATH=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/grpc/build/third_party/protobuf/cmake/protobuf/ \
cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT" -DBUILD_RVS=ON ..

#cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT" ..
#cmake  -DGRPC_ROOT="$GRPC_ROOT" -DBUILD_STANDALONE=off -DBUILD_RVS=Off .. 

#if [[ -z $? ]] ; then make -j $(nproc) install ; fi
make -j$(nproc)
popd

RDC_LIB_DIR=/opt/rocm/rdc/lib
GRPC_LIB_DIR=/opt/grpc/lib
echo -e "${GRPC_LIB_DIR}\n${GRPC_LIB_DIR}64" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
echo -e "${RDC_LIB_DIR}\n${RDC_LIB_DIR}64" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
ldconfig

set -x 

OPTION_BUILD_RVS_SUPPORT=1
OPTION_CLEAN_BUILD_RDC=1
OPTION_CREATE_PKG=1
GRPC_PROTOC_ROOT=/opt/grpc

apt install -y automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl
apt install -y libyaml-cpp-dev libabsl-dev

CMAKE_PARAMS="-DROCM_DIR=/opt/rocm -DGRPC_ROOT=$GRPC_PROTOC_ROOT -DCMAKE_INSTALL_PREFIX=/opt/rocm" 

if [[ $OPTION_CREATE_PKG == 1 ]] ;          then CMAKE_PARAMS="$CMAKE_PARAMS -DCPACK_GENERATOR=DEB" ; fi
if [[ $OPTION_BUILD_RVS_SUPPORT == 1 ]] ;   then apt install rocm-validation-suite -y ; CMAKE_PARAMS="$CMAKE_PARAMS -DBUILD_RVS=ON" ; fi

if [[ ! -z rdc ]] ; then
    git clone https://github.com/ROCm/rdc
fi
pushd rdc
pwd
mkdir -p build
cd build

if [[ $OPTION_CLEAN_BUILD_RDC -eq 1 ]] ; then rm -rf build ; fi


CMAKE_PREFIX_PATH=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/grpc/build/third_party/protobuf/cmake/protobuf/ \
cmake $CMAKE_PARAMS ..

#cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT" ..
#cmake  -DGRPC_ROOT="$GRPC_ROOT" -DBUILD_STANDALONE=off -DBUILD_RVS=Off .. 

#if [[ -z $? ]] ; then make -j $(nproc) install ; fi
make -j$(nproc) install package
popd

RDC_LIB_DIR=/opt/rocm/rdc/lib
GRPC_LIB_DIR=/opt/grpc/lib
echo -e "${GRPC_LIB_DIR}\n${GRPC_LIB_DIR}64" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
echo -e "${RDC_LIB_DIR}\n${RDC_LIB_DIR}64" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
ldconfig

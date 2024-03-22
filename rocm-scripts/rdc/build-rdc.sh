apt install -y automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl
apt install -y libyaml-cpp-dev
set -x 
if [[ ! -z rdc ]] ; then
    git clone https://github.com/ROCm/rdc
fi
pushd rdc
pwd
mkdir -p build
cd build
rm -rf ./*

# default installation location is /opt/rocm, specify with -DROCM_DIR or -DCMAKE_INSTALL_PREFIX
CMAKE_PREFIX_PATH=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/grpc/build/third_party/protobuf/cmake/protobuf/ cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT" ..

#cmake -B build -DGRPC_ROOT="$GRPC_ROOT" -DBUILD_RVS=ON ..
#cmake  -DGRPC_ROOT="$GRPC_ROOT" -DBUILD_STANDALONE=off -DBUILD_RVS=Off .. 

if [[ -z $? ]] ; then make -j $(nproc) install ; fi
popd

set -x 
#centos : gcrp part working.
#ubuntu: in test.
apt install doxygen  libcap-dev
build_grpc=0
build_rdc=1

if [[ ! -d grpc ]] ; then 
    git clone -b v1.61.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
fi

pushd  grpc
export GRPC_ROOT=/opt/grpc
cmake -B build \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=Release
make -C build -j $(nproc)
sudo make -C build install
echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf
popd


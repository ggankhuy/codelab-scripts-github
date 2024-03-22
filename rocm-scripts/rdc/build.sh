
#git clone -b v1.61.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
cd grpc
export GRPC_ROOT=/opt/grpc
cmake -B build \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=Release
make -C build -j $(nproc)
sudo make -C build install
echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf

#git clone https://github.com/ROCm/rdc
cd rdc
mkdir -p build
# default installation location is /opt/rocm, specify with -DROCM_DIR or -DCMAKE_INSTALL_PREFIX
cmake -B build -DGRPC_ROOT="$GRPC_ROOT"
make -C build -j $(nproc)
make -C build install

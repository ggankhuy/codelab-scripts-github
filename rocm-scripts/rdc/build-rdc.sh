set -x 
pushd rdc
mkdir -p build
rm -rf ./*
# default installation location is /opt/rocm, specify with -DROCM_DIR or -DCMAKE_INSTALL_PREFIX
cmake -B build -DGRPC_ROOT="$GRPC_ROOT" -DBUILD_RVS=ON
make -C build -j $(nproc)
make -C build install
popd

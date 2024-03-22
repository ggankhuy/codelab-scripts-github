set -x
#git clone https://github.com/ROCm/rdc
cd rdc
mkdir -p build
# default installation location is /opt/rocm, specify with -DROCM_DIR or -DCMAKE_INSTALL_PREFIX
cmake -B build -DGRPC_ROOT="$GRPC_ROOT"
make -C build -j $(nproc)
make -C build install

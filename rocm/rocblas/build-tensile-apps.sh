FILENAME=tensile_dat_reader
#hipcc  -std=c++17 -I/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/Tensile/Source/lib/include/  $FILENAME.cpp
#hipcc  -I/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/Tensile/Source/lib/include/ --offload-arch=gfx908 --save-temps -c $FILENAME.cpp
hipcc  -std=c++17 -I/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/Tensile/Source/lib/include/  -c $FILENAME.cpp
hipcc $FILENAME.o  -o /opt/rocm/lib/librocblas.so -o $FILENAME.out

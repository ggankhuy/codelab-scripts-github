# build success with following environment:
# rocm5.2 installed (rocm5.4 was OK)
# Tensile source code in ~/gg/git/, once cloned checkout using tensile tag (from 5.2): 9ca08f38c4c3bfe6dfa02233637e7e3758c7b6db
# Checkout using tensile_tag with later than 5.2 will break (i.e.5.4).
# 
FILENAME=tensile_dat_reader
FILENAME=tensile_dat_reader_simple
TENSILE_ROOT=/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/Tensile/Source/lib/include/
TENSILE_ROOT=/root/gg/git/Tensile
TENSILE_ROOT=/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile

#hipcc  -std=c++17 -I/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/Tensile/Source/lib/include/  $FILENAME.cpp
#hipcc  -I/root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/Tensile/Source/lib/include/ --offload-arch=gfx908 --save-temps -c $FILENAME.cpp
hipcc  -std=c++17 -I$TENSILE_ROOT/Tensile/Source/lib/include/  -c $FILENAME.cpp $TENSILE_ROOT/Tensile/Source/lib/source/Tensile.cpp
hipcc $FILENAME.o Tensile.o  -o /opt/rocm/lib/librocblas.so -o $FILENAME.out
hipcc  -std=c++17 -g -I$TENSILE_ROOT/Tensile/Source/lib/include/  -c $FILENAME.cpp $TENSILE_ROOT/Tensile/Source/lib/source/Tensile.cpp
hipcc $FILENAME.o Tensile.o  -o /opt/rocm/lib/librocblas.so -o $FILENAME.dbg.out

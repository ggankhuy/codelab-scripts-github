#/opt/rocm-4.5.2/hip/include/hip/
#-I$(ROCM_PATH)/hsa/include
rm -rf build/*
mkdir build
cd build
FILENAME=sscal
FILENAME=example_sgemm
FILENAME=gemm
ln -s ../$FILENAME.cpp
for i in helpers.hpp ArgParser.hpp ; do 
    ln -s ../$i .
done

hipcc --offload-arch=gfx908 --save-temps -c $FILENAME.cpp
hipcc $FILENAME.o  /opt/rocm-4.5.2/lib/librocblas.so
cd ..


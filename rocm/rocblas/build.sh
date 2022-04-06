#/opt/rocm-4.5.2/hip/include/hip/
#-I$(ROCM_PATH)/hsa/include
mkdir build
cd build
ln -s ../sscal.cpp
hipcc --offload-arch=gfx908 --save-temps -c sscal.cpp
hipcc sscal.o  /opt/rocm-4.5.2/lib/librocblas.so
cd ..


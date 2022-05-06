# note that this is not building. may try 5.0 or 5.1?
#/opt/rocm-4.5.2/hip/include/hip/
#-I$(ROCM_PATH)/hsa/include
rm -rf build/*
mkdir build
cd build
FILENAME=example-c
ln -s ../$FILENAME.c
#for i in helpers.hpp ArgParser.hpp ; do 
#    ln -s ../$i .
#done

hipcc --offload-arch=gfx908 --save-temps -c $FILENAME.c -I/opt/rocm-4.5.2/include/
hipcc $FILENAME.o  /opt/rocm-4.5.2/lib/librocblas.so
cd ..


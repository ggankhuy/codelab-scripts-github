# does not build bad!!!

# 1. there are other get_time telemetry api defined in client/ directory in utility.cpp/h 
# but if anyone wants to use, it will get into extra complication.

# 2. hipblas_init is defined in not library/src, but in client/ directory.
# but hipblas rpm package does not have binary library in client. 
# therefore, simple include of libhipblas.so will not suffice just to build even main hello world
# application where hipblas_init apparently initializes entry data.

# 3. User may be able to chug along by moving directories here and there specially from client directory but 
# this is a messy operation and defeats to purpose of having hipblas API.

# tensor_ops: builds ok with tensor.hpp/gpu_mem.hpp softlink.

mkdir build
cd build
#FILENAME=testing_axpby
#FILENAME=example_axpyi
FILENAME=miopen-test
FILENAME=tensor_ops
#FILENAME=tensor_ops-2
ln -s ../$FILENAME.cpp
#ln -s ../testing.hpp .
ln -s ../tensor.hpp .
ln -s ../gpu_mem.hpp .
hipcc -c $FILENAME.cpp -I/git/codelab/gpu/rocm/miopen -I/root/ROCm-4.5/MIOpen/src/include/ -std=c++14 -I/root/ROCm-4.5/MIOpen/test
#hipcc -c $FILENAME.cpp -I/git/codelab/gpu/rocm/miopen -std=c++14
#hipcc $FILENAME.o /opt/rocm-4.5.2/lib/libMIOpen.so
hipcc $FILENAME.o /opt/rocm-4.5.2/lib/libMIOpen.so.1 /usr/lib64/libboost_filesystem.so


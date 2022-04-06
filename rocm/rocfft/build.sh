# does not build bad!!!

# 1. there are other get_time telemetry api defined in client/ directory in utility.cpp/h 
# but if anyone wants to use, it will get into extra complication.

# 2. hipblas_init is defined in not library/src, but in client/ directory.
# but hipblas rpm package does not have binary library in client. 
# therefore, simple include of libhipblas.so will not suffice just to build even main hello world
# application where hipblas_init apparently initializes entry data.

# 3. User may be able to chug along by moving directories here and there specially from client directory but 
# this is a messy operation and defeats to purpose of having hipblas API.

mkdir build
cd build
#FILENAME=testing_axpby
FILENAME=rocfft_example_callback
ln -s ../$FILENAME.cpp
#ln -s ../testing.hpp .
hipcc -c --save-temps --offload-arch=gfx908 $FILENAME.cpp
hipcc $FILENAME.o /opt/rocm-4.5.2/lib/librocfft.so.0


#/opt/rocm-4.5.2/hip/include/hip/
#-I$(ROCM_PATH)/hsa/include
hipcc -c sscal.cpp
hipcc sscal.o  /opt/rocm-4.5.2/lib/librocblas.so


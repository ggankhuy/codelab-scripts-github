g++ -I /opt/rocm-5.7.0/roctracer/include/ main.c /opt/rocm-5.7.0/lib/libroctracer64.so
#hipcc -I /opt/rocm-5.7.0/roctracer/include/ p61.cpp /opt/rocm-5.7.0/lib/libroctracer64.so
#hipcc p61.cpp

#g++ -I /opt/rocm-5.7.0/roctracer/include/ main.c /opt/rocm-5.7.0/lib/libroctracer64.so

# build with tracer support
#hipcc -I /opt/rocm-5.7.0/roctracer/include/ p61.cpp /opt/rocm-5.7.0/lib/libroctracer64.so

# build with both roctx + rocracer
#hipcc -I /opt/rocm-5.7.0/roctracer/include/ p61.cpp /opt/rocm-5.7.0/lib/libroctracer64.so  /opt/rocm-5.7.0/lib/libroctx64.so
hipcc -I/opt/rocm-5.7.0/include -L/opt/rocm-5.7.0/lib -lroctracer64 p61.cpp
rocprof --trace-start off --hip-trace ./a.out

# build with roctx only
#hipcc -I /opt/rocm-5.7.0/roctracer/include/ p61.cpp /opt/rocm-5.7.0/lib/libroctx64.so
#hipcc -I/opt/rocm-5.7.0/include -L/opt/rocm-5.7.0/lib -lroctracer64 -lroctx64 p61.cpp
rocprof --hip-trace --roctx-trace ./a.out

# build without any tracer (working)
#hipcc p61.cpp

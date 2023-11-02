g++ -I /opt/rocm-5.7.0/roctracer/include/ main.c /opt/rocm-5.7.0/lib/libroctracer64.so

# build with tracer support
#hipcc -I /opt/rocm-5.7.0/roctracer/include/ p61.cpp /opt/rocm-5.7.0/lib/libroctracer64.so

# build with both roctx + rocracer
#hipcc -I /opt/rocm-5.7.0/roctracer/include/ p61.cpp /opt/rocm-5.7.0/lib/libroctracer64.so  /opt/rocm-5.7.0/lib/libroctx64.so

# build with roctx only
hipcc -I /opt/rocm-5.7.0/roctracer/include/ p61.cpp /opt/rocm-5.7.0/lib/libroctx64.so

# build without any tracer (working)
#hipcc p61.cpp

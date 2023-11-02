g++ -I /opt/rocm/roctracer/include/ main.c /opt/rocm/lib/libroctracer64.so

# build with tracer support
#hipcc -I /opt/rocm/roctracer/include/ p61.cpp /opt/rocm/lib/libroctracer64.so

# build with both roctx + rocracer
#hipcc -I /opt/rocm/roctracer/include/ p61.cpp /opt/rocm/lib/libroctracer64.so  /opt/rocm/lib/libroctx64.so

# build with roctx only
hipcc -I /opt/rocm/roctracer/include/ p61.cpp /opt/rocm/lib/libroctx64.so

# build without any tracer (working)
#hipcc p61.cpp

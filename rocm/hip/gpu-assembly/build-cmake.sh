rocm_path=/opt/rocm
rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"

for i in project_name in vector vector-1024 ; do
    mkdir ./log
    mkdir build
    cd build
    rm -rf ./*
    #cmake -DCMAKE_PREFIX_PATH=/opt/rocm/hip/lib .. 2>&1 | tee ./log/cmake.log/
    #make 2>&1 | tee ./log/make.log
    CXX=${rocm_path}/bin/$compiler ${cmake_executable} \
        -DCMAKE_PREFIX_PATH="${rocm_path}/hip/lib" \
        -DCMAKE_MODULE_PATH="${rocm_path}/hip/cmake" \
        -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" \
        -DROCM_PATH=${rocm_path} \
        -DPROJECT_NAME=${project_name}
        ..
        make
    cd ..
done


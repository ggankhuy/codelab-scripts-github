rocm_path=/opt/rocm
rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"
project_name=vector

for sub_project_name in vector vector-4 ; do
    echo "Generating project: $project_name..."
    BUILD_DIR=build-$sub_project_name
    mkdir ./log
    mkdir $BUILD_DIR
    mkdir bindir
    cd $BUILD_DIR
    rm -rf ./*
    #cmake -DCMAKE_PREFIX_PATH=/opt/rocm/hip/lib .. 2>&1 | tee ./log/cmake.log/
    #make 2>&1 | tee ./log/make.log
    CXX=${rocm_path}/bin/$compiler ${cmake_executable} \
        -DCMAKE_PREFIX_PATH="${rocm_path}/hip/lib" \
        -DCMAKE_MODULE_PATH="${rocm_path}/hip/cmake" \
        -DCMAKE_SHARED_LINKER_FLAGS="${rocm_rpath}" \
        -DROCM_PATH=${rocm_path} \
        -DPROJECT_NAME=${project_name} \
        -DSUB_PROJECT_NAME=$sub_project_name \
        .. 2>&1 | tee $BUILD_DIR/cmake-stdout.log
    make
    cp ${project_name} ../bindir/${sub_project_name}
    cd ..
done


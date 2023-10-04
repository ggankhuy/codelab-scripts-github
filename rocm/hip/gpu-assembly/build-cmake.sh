rocm_path=/opt/rocm
rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"
project_name=vector

LOG_FILE_BUILD=cmake-stdout.log
LOG_FILE_EXEC=exec.log

#for sub_project_name in vector vector-4 ; do
rm -rf bindir/*

for sub_project_name in vector vector4 vector1024 ; do
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
        .. 2>&1 | tee -a ./$LOG_FILE_BUILD
    sleep 4
    make 2>&1  | tee -a ./$LOG_FILE_BUILD
    cp ${project_name} ../bindir/${sub_project_name}
    ./${project_name} | tee -a ./$LOG_FILE_EXEC
    cd ..
done
tree bindir


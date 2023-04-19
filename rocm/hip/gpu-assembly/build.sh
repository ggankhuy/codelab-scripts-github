rocm_path=/opt/rocm
rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"
project_name=vector

LOG_FILE_BUILD=cmake-stdout.log
LOG_FILE_EXEC=exec.log

#for sub_project_name in vector vector-4 ; do
rm -rf bindir/*

#for sub_project_name in vector vector4 vector64 vector1024 matrix16x16; do
for sub_project_name in vector ; do

    for envvar in OP DATATYPE DATASHAPE ; do
        unset $envvar
    done
    echo "-----------------"
    case "$sub_project_name" in
        "matrix16x16")
            echo "Setting export variables for matrix16x16"
            export DATATYPE=int
            export DATASHAPE=matrix
            ;;
        "vector")
            echo "Setting export variables for vector"
            export OP=add
            ;;
        "vector4")
            echo "Setting export variables for vector4"
            export OP=mul
            ;;
        "vector1024")
            echo "Setting export variables for vector1024"
            export OP=mul_add
            
            ;;
    esac

    echo "Generating project: $project_name..."
    BUILD_DIR=build-$sub_project_name
    mkdir ./log
    mkdir $BUILD_DIR
    mkdir ./bindir
    sudo rm -rf ./bindir/*
    rm -rf $BUILD_DIR/*
    cd $BUILD_DIR
    sudo ln -s ../vector.cpp .
    sudo ln -s ../vector.h .

    export PROJECT_NAME=$sub_project_name
    CMD="hipcc --save-temps -DOPT_SUB_PROJECT_NAME=$sub_project_name $project_name.cpp -o $project_name"
    echo $CMD
    $CMD

    ln -s `pwd`/${project_name} ../bindir/${sub_project_name}
    ../bindir/${sub_project_name} | tee -a ./$LOG_FILE_EXEC
    cd ..

done

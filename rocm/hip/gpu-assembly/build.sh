rocm_path=/opt/rocm
rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"
project_name=vector

LOG_FILE_BUILD=cmake-stdout.log
LOG_FILE_EXEC=exec.log

#for sub_project_name in vector vector-4 ; do
rm -rf bindir/*

#for sub_project_name in vector ; do
for sub_project_name in vector vector4 vector64 vector1024 matrix_32x32_8x8x1 matrix_256x256_32x32x1; do

#    if (env_project_name_str == "matrix_32x32_8x8x1") { MAT_X=32; MAT_Y=32; N=(MAT_X*MAT_Y; T_X=8; T_Y=8; T_Z=1; }
#    if (env_project_name_str == "matrix_32x32_4x4x1") { MAT_X=32; MAT_Y=32; N=(MAT_X*MAT_Y; T_X=4; T_Y=4; T_Z=1;  }
#    if (env_project_name_str == "matrix_256x256_32x32x1") { MAT_X=16; MAT_Y=64; N=(MAT_X*MAT_Y; T_X=32; T_Y=32; T_Z=1; }
#    if (env_project_name_str == "matrix_256x256_64x16x1") { MAT_X=256; MAT_Y=256; N=(MAT_X*MAT_Y; T_X=64; T_Y=16; T_Z=1; }
#    if (env_project_name_str == "matrix_256x256_16x64x1") { MAT_X=256; MAT_Y=256; N=(MAT_X*MAT_Y; T_X=16; T_Y=64; T_Z=1; }

    for envvar in OP DATATYPE DATASHAPE ; do
        unset $envvar
    done
    echo "-----------------"
    case "$sub_project_name" in
        "matrix_32x32_8x8x1")
            echo "Setting export variables for 32x32_8x8x1"
            export FILENAME=matrix

            # unused for now.
            export DATATYPE=int
            export DATASHAPE=matrix
            ;;
        "matrix_256x256_32x32x1")
            echo "Setting export variables for 32x32_8x8x1"
            export FILENAME=matrix
            ;;
        "vector")
            echo "Setting export variables for vector"
            export FILENAME=vector
            export OP=add
            ;;
        "vector4")
            echo "Setting export variables for vector4"
            export FILENAME=vector
            export OP=mul
            ;;
        "vector1024")
            echo "Setting export variables for vector1024"
            export FILENAME=vector
            export OP=mul_add
            
            ;;
    esac

    echo "Generating project: $project_name..."
    BUILD_DIR=build-$sub_project_name
    mkdir ./log
    mkdir $BUILD_DIR
    mkdir ./bindir
    #sudo rm -rf ./bindir/*
    #rm -rf $BUILD_DIR/*
    cd $BUILD_DIR
    sudo ln -s ../$FILENAME.cpp .
    sudo ln -s ../$FILENAME.h .

    export PROJECT_NAME=$sub_project_name
    CMD="hipcc --save-temps -DOPT_SUB_PROJECT_NAME=$sub_project_name $FILENAME.cpp -o $sub_project_name"
    echo $CMD
    $CMD

    echo ln -s `pwd`/${sub_project_name} ../bindir/${sub_project_name}
    ln -s `pwd`/${sub_project_name} ../bindir/${sub_project_name}
    AMD_LOG_LEVEL=4 ../bindir/${sub_project_name} | tee -a ./$LOG_FILE_EXEC
    cd ..
    ls -l bindir
done
tree -fs bindir

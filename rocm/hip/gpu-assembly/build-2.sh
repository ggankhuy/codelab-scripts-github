rocm_path=/opt/rocm
CMP_ARGS_FILENAME_SRC=cmp_args.txt
CMP_ARGS_FILENAME_SH=cmp_args.sh
CMP_ARGS_FILENAME_C_HDR=cmp_args.h

cat $CMP_ARGS_FILENAME_SRC | tee -a $CMP_ARGS_FILENAME_SH
echo "" | tee $CMP_ARGS_FILENAME_C_HDR
while IFS= read -r line
do
    VAR=`echo "$line" | tr -s ' ' | cut -d '=' -f1`
    VALUE=`echo "$line" | tr -s ' ' | cut -d '=' -f2`
    if [[ ! -z $VAR ]] && [[ ! -z $VALUE ]] ; then 
        echo "#define $VAR $VALUE" | tee -a $CMP_ARGS_FILENAME_C_HDR 
    else
        echo "" | tee -a $CMP_ARGS_FILENAME_C_HDR

    fi
done < "$CMP_ARGS_FILENAME_SRC"

source $CMP_ARGS_FILENAME_SH

rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"
project_name=vector

LOG_FILE_BUILD=cmake-stdout.log
LOG_FILE_EXEC=exec.log

#for sub_project_name in vector vector-4 ; do
rm -rf bindir/*

for sub_project_name in vector matrix_256x256_32x32x1 matrix_256x256_32x32x1_float ; do
#for sub_project_name in vector vector1024MA; do
#for sub_project_name in vector vector4 vector64 vector1024 matrix_32x32_8x8x1 matrix_256x256_32x32x1 matrix_256x256_32x32x1_float; do

#    if (env_project_name_str == "matrix_32x32_8x8x1") { MAT_X=32; MAT_Y=32; N=(MAT_X*MAT_Y; T_X=8; T_Y=8; T_Z=1; }
#    if (env_project_name_str == "matrix_32x32_4x4x1") { MAT_X=32; MAT_Y=32; N=(MAT_X*MAT_Y; T_X=4; T_Y=4; T_Z=1;  }
#    if (env_project_name_str == "matrix_256x256_32x32x1") { MAT_X=16; MAT_Y=64; N=(MAT_X*MAT_Y; T_X=32; T_Y=32; T_Z=1; }
#    if (env_project_name_str == "matrix_256x256_64x16x1") { MAT_X=256; MAT_Y=256; N=(MAT_X*MAT_Y; T_X=64; T_Y=16; T_Z=1; }
#    if (env_project_name_str == "matrix_256x256_16x64x1") { MAT_X=256; MAT_Y=256; N=(MAT_X*MAT_Y; T_X=16; T_Y=64; T_Z=1; }

    echo "-----------------"
    COMPILER_ARGS=" "
    case "$sub_project_name" in
        "matrix_32x32_8x8x1")
            echo "Setting export variables for 32x32_8x8x1"
            FILENAME=matrix
            COMPILER_ARGS=$COMPILER_ARGS" -DOP=$ARG_OP_ADD"
            COMPILER_ARGS=$COMPILER_ARGS" -DDATATYPE=$ARG_DATATYPE_INT32"
            ;;
        "matrix_256x256_32x32x1")
            echo "Setting export variables for 32x32_8x8x1"
            FILENAME=matrix
            COMPILER_ARGS=$COMPILER_ARGS" -DOP=$ARG_OP_ADD"
            COMPILER_ARGS=$COMPILER_ARGS" -DDATATYPE=$ARG_DATATYPE_INT32"
            ;;
        "matrix_256x256_32x32x1_float")
            echo "Setting export variables for 32x32_8x8x1"
            FILENAME=matrix
            COMPILER_ARGS=$COMPILER_ARGS" -DOP=$ARG_OP_ADD"
            COMPILER_ARGS=$COMPILER_ARGS" -DDATATYPE=$ARG_DATATYPE_FP32"
            ;;
        "vector")
            echo "Setting export variables for vector"
            FILENAME=vector
            COMPILER_ARGS=$COMPILER_ARGS" -DOP=$ARG_OP_ADD"
            export OP=add
            ;;
        "vector4M")
            echo "Setting export variables for vector4"
            FILENAME=vector
            COMPILER_ARGS=$COMPILER_ARGS" -DOP=$ARG_OP_MUL"
            COMPILER_ARGS=$COMPILER_ARGS" -DSIZE=$ARG_DATASIZE_VEC_4"
            ;;
        "vector1024MA")
            echo "Setting export variables for vector1024"
            FILENAME=vector
            COMPILER_ARGS=$COMPILER_ARGS" -DOP=$ARG_OP_MUL_ADD"
            COMPILER_ARGS=$COMPILER_ARGS" -DSIZE=$ARG_DATASIZE_VEC_1024"
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
    sudo ln -s ../$CMP_ARGS_FILENAME_C_HDR .

    export PROJECT_NAME=$sub_project_name

    CMD="hipcc --save-temps $COMPILER_ARGS $FILENAME.cpp -o $sub_project_name"
    echo "$CMD"
    $CMD

    echo ln -s `pwd`/${sub_project_name} ../bindir/${sub_project_name}
    ln -s `pwd`/${sub_project_name} ../bindir/${sub_project_name}
    ../bindir/${sub_project_name} | tee -a ./$LOG_FILE_EXEC
    cd ..
    ls -l bindir
done
tree -fs bindir

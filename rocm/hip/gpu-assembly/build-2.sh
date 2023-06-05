#!/bin/bash

rocm_path=/opt/rocm

# Process compiler arguments defines, create source sh and header file.

CMP_ARGS_FILENAME_SRC=cmp_args_2.txt
CMP_ARGS_FILENAME_SH=cmp_args.sh
CMP_ARGS_FILENAME_C_HDR=cmp_args.h
SUMMARY_LOG=summary.log

cat $CMP_ARGS_FILENAME_SRC > $CMP_ARGS_FILENAME_SH
echo "" | tee $CMP_ARGS_FILENAME_C_HDR
while IFS= read -r line
do
    VAR=`echo "$line" | tr -s ' ' | cut -d '=' -f1`
    VALUE=`echo "$line" | tr -s ' ' | cut -d '=' -f2`
    if [[ ! -z $VAR ]] && [[ ! -z $VALUE ]] ; then 
        echo "#define $VAR $VALUE" >> $CMP_ARGS_FILENAME_C_HDR 
    else
        echo "" >> $CMP_ARGS_FILENAME_C_HDR

    fi
done < "$CMP_ARGS_FILENAME_SRC"


source $CMP_ARGS_FILENAME_SH
SLEEP=0

# This script defines.

rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"
project_name=vector

LOG_FILE_BUILD=cmake-stdout.log
LOG_FILE_EXEC=exec.log

rm -rf bindir/*
echo -ne "" | tee $SUMMARY_LOG

# Create executables through loop.

FILENAME=tensor

    #ADD_INT32_4 \
    #ADD_INT32_1024 \
    #ADD_FP32_16_16; do
for EXEC_NAME_PART in \
    ADD_INT32_8 \
    ADD_INT32_64 \
    ADD_INT32_16_16 \
    ;do

    echo "========================"
    TOKEN_OP=`echo $EXEC_NAME_PART | tr -s ' '  |cut -d '_' -f1`
    TOKEN_DATATYPE=`echo $EXEC_NAME_PART | tr -s ' '  |cut -d '_' -f2`
    TOKEN_X=`echo $EXEC_NAME_PART | tr -s ' '  |cut -d '_' -f3`
    TOKEN_Y=`echo $EXEC_NAME_PART | tr -s ' '  |cut -d '_' -f4`
    TOKEN_Z=`echo $EXEC_NAME_PART | tr -s ' '  |cut -d '_' -f5`

    if TOKEN_Y="" ; then TOKEN_Y=1 ; fi
    if TOKEN_Z="" ; then TOKEN_Z=1 ; fi

    ARG1="-DOP=$TOKEN_OP"
    ARG2="-DDATATYPE=$TOKEN_DATATYPE"
    ARG3="-DX=$TOKEN_X"
    ARG4="-DY=$TOKEN_Y"
    ARG5="-DZ=$TOKEN_Z"

    COMPILER_ARGS=" $ARG1 $ARG2 $ARG3 $ARG4 $ARG5"
    echo "args: $ARGS"

    # tiles (workgroup)
    # 4x4, 8x8, 16x16, 32x32, 16x64, 64x16
    # inside vector. 

    for tile in 4x1 8x1 4x4 8x8; do
        echo "------------------------"
        TILE_X=`echo $tile | tr -s ' '  |cut -d 'x' -f1`
        TILE_Y=`echo $tile | tr -s ' '  |cut -d 'x' -f2`

        TILE_ARG="-DTILEX=$TILE_X -DTILEY=$TILE_Y"

        BUILD_DIR=build-$EXEC_NAME_PART-$tile
        EXEC_NAME_FULL=$BUILD_DIR
        echo "Generating project: $BUILD_DIR"
        echo "tokens: $TOKEN_X $TOKEN_Y $TOKEN_Z, tiles: $TILE_X $TILE_Y"

        if [[ $TILE_X -gt $TOKEN_X ]] ; then 
            echo "1. tile X=$TILE_X is greater than X-dim=$TOKEN_X of tensor. This build will skip." ; 
            sleep $SLEEP
            echo "$BUILD_DIR: BYPASS" | tee -a $SUMMARY_LOG
            continue 
        fi

        if [[ $TILE_Y -gt $TOKEN_Y ]] ; then 
            echo "2. tile Y=$TILE_Y is greater than Y-dim=$TOKEN_Y of tensor. This build will skip." ; 
            sleep $SLEEP 
            echo "$BUILD_DIR: BYPASS" | tee -a $SUMMARY_LOG
            continue 
        fi

        mkdir ./log
        mkdir $BUILD_DIR
        mkdir ./bindir
        #sudo rm -rf ./bindir/*
        #rm -rf $BUILD_DIR/*
        cd $BUILD_DIR
        sudo ln -s ../$FILENAME.cpp .
        sudo ln -s ../$FILENAME.h .
        sudo ln -s ../$CMP_ARGS_FILENAME_C_HDR .

        export PROJECT_NAME=$EXEC_NAME_PART

        CMD="hipcc --save-temps $COMPILER_ARGS $TILE_ARG $FILENAME.cpp -o $EXEC_NAME_FULL"
        echo "$CMD" 2>&1 | tee build.log
        $CMD 2>&1 | tee -a build.log

        echo ln -s `pwd`/${EXEC_NAME_FULL} ../bindir/${EXEC_NAME_FULL}
        ln -s `pwd`/${EXEC_NAME_FULL} ../bindir/${EXEC_NAME_FULL}
        ../bindir/${EXEC_NAME_FULL} | tee -a ./$LOG_FILE_EXEC
        cd ..
        echo "$BUILD_DIR: OK" | tee -a $SUMMARY_LOG
        done
done
tree -fs bindir

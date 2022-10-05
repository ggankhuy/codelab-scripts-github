source sh/common.sh

for var in "$@"
do
    if [[ $var == *"comp="* ]]  ; then
        comp=`echo $var | cut -d '=' -f2`
        COMP=$comp
        COMP_OLD=$comp
        echo COMP old: $COMP
        COMP=$(echo $COMP | sed "s/-/_/g")
        echo COMP new: $COMP
    fi
done

function llvm() {
    CURR_BUILD=llvm-project
    build_entry $CURR_BUILD
    pushd $CURR_BUILD
    mkdir build ; cd build
    echo "cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$VERSION.$MINOR_VERSION/llvm -DCMAKE_BUILD_TYPE=Release"
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$VERSION.$MINOR_VERSION/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC 2>&1 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    popd
}

function f3 () {
    i=$1
    build_entry $i
    CURR_BUILD=$i
    pwd 
    #pushd $CURR_BUILD
    pushd $COMP_OLD
    mkdir build; cd build
    cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm/ .. 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
}

function rocminfo()  {
    f3 rocminfo
}

function MIOpenGEMM() {
    f3 MIOpenGEMM
}

function ROCm_Device_Lib() {
    setup_root_rocm_softlink
    CURR_BUILD=ROCm-Device-Libs
    build_entry $CURR_BUILD
    DEVICE_LIBS=$ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir -p "$DEVICE_LIBS/build"
    cd "$DEVICE_LIBS/build"
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" -DCMAKE_INSTALL_PREFIX=/opt/rocm/ .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    cp *deb *rpm $CONFIG_BUILD_PKGS_LOC/
}

function ROCm_CompilerSupport() {
    CURR_BUILD=ROCm-CompilerSupport
    build_entry $CURR_BUILD
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    cd lib/comgr/
    LLVM_PROJECT=$ROCM_SRC_FOLDER/llvm-project
    COMGR=$ROCM_SRC_FOLDER/$CURR_BUILD/lib/comgr
}


function ROCmValidationSuite() {
    CURR_BUILD=ROCmValidationSuite
    pushd $ROCM_SRC_FOLDER/ROCmValidationSuite
    mkdir build ; cd build
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-1" >> $LOG_SUMMARY ; fi
    cmake ..  2>&1 | tee -a $LOG_DIR/$CURR_BUILD-1.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-2" >> $LOG_SUMMARY ; fi
    make -j`nproc` | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make install | tee -a $LOG_DIR/$CURR_BUILD-3.log
    popd
}

function ROCT_Thunk_Interface() {
    f3 ROCT-Thunk-Interface
}

function rocm_bandwidth_test() {
    f3 rocm_bandwidth_test
}

function rocm_cmake () {
    f3 rocm_cmake
}

function rocm_smi_lib () {
    f3 rocm_smi_lib
}

function rocprofiler() {
    f3 rocprofiler
}

function COMGR() {
    CURR_BUILD=ROCm-CompilerSupport
    build_entry $CURR_BUILD
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    cd lib/comgr/
    LLVM_PROJECT=$ROCM_SRC_FOLDER/llvm-project
    COMGR=$ROCM_SRC_FOLDER/$CURR_BUILD/lib/comgr

    CURR_BUILD=COMGR
    build_entry $CURR_BUILD
    mkdir -p "$COMGR/build"
    cd "$COMGR/build"

    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" .. | tee  -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET  2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make test 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
}

function protobuf() {
    CURR_BUILD=protobuf
    build_entry $CURR_BUILD

    git clone https://github.com/protocolbuffers/protobuf.git
    cd $CURR_BUILD 
    git checkout v3.16.0
    git submodule update --init --recursive
    mkdir build ; cd build
    cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    make -j`nproc` 
    make $INSTALL_TARGET    
    cd ../..

}

function AMDMIGraphX() {
    #./tools/install_prereqs.sh

    CURR_BUILD=AMDMIGraphX
    # following commented lines replaced by install_prereqs.sh, hopefully, intest.
    pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz | tee  $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    rbuild build -d depend --cxx=/opt/rocm/llvm/bin/clang++ | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    mkdir build; cd build
    CXX=/opt/rocm/llvm/bin/clang++ cmake .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    popd

}

function rocBLAS() {
    i=rocBLAS
    CURR_BUILD=$i
    build_entry $i
    #patch_rocblas $base_dir_this_script/rocBLAS/cmake/ $base_dir_api 
    #cat rocBLAS/cmake/virtualenv.cmake  | grep upgrade -i | tee $LOG_DIR/$CURR_BUILD.log
    popd
    pushd $ROCM_SRC_FOLDER/$i
#   ./install.sh -icd --logic asm_full | tee $LOG_DIR/$CURR_BUILD.log
    ./install.sh -icd --no-tensile --logic asm_full | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
}

function MIOpen() {
    i=MIOpen
    CURR_BUILD=$i
    build_entry $i
    pushd $i
    cmake -P install_deps.cmake --minimum | tee $LOG_DIR/$CURR_BUILD-1.log
    mkdir build; cd build
    rm -rf ./*
    echo "current dir: "  | tee $LOG_DIR/$CURR_BUILD-2.log
    pwd 2>&1 | tee $LOG_DIR/$CURR_BUILD-2.log
    CXX=/opt/rocm/llvm/bin/clang++ cmake -DMIOPEN_BACKEND=HIP -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ .. 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 1." >> $LOG_SUMMARY ; fi

    echo "make -j$NPROC $BUILD_TARGET 2>&1" | tee $LOG_DIR/$CURR_BUILD-3.log
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-3.log

    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 2." >> $LOG_SUMMARY ; fi
    echo "make $INSTALL_TARGET 2>&1" | tee $LOG_DIR/$CURR_BUILD-4.log

    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-4.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 3." >> $LOG_SUMMARY ; fi
    popd
}

function hipAMD() {
    CURR_BUILD=hipamd
    build_entry $CURR_BUILD
    cd $ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir build ; cd build

    pushd ../..
    export HIPAMD_DIR="$(readlink -f hipamd)"
    export HIP_DIR="$(readlink -f hip)"
    export ROCclr_DIR="$(readlink -f ROCclr)"
    export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
    echo HIPAMD_DIR: $HIPAMD_DIR, HIP_DIR: $HIP_DIR, ROCclr_DIR: $ROCclr_DIR, OPENCL_DIR: $OPENCL_DIR
    popd
    sudo ln -s $ROCM_SRC_FOLDER/HIP $ROCM_SRC_FOLDER/hip
    #cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_P$
    cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" .. 2>&1 | tee $LOG$
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    make -j$NPROC 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    #make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

}


function f1() {
    i=$1
    CURR_BUILD=$i
    build_entry $i
    pushd $ROCM_SRC_FOLDER/$i
    pwd
    if [[ $i == "rocSOLVER" ]] ; then
        ./install.sh -i 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    else
        ./install.sh -icd 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    fi
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    popd
}

function rocPRIM() {
    f2 rocPRIM
}
function hipCUB() {
    f2 hipCUB
}

function f2() {
    i=$1
    CURR_BUILD=$i
    build_entry $i
    pushd $ROCM_SRC_FOLDER/$i
    mkdir build; cd build
    CXX=/opt/rocm/hip/bin/hipcc cmake -DBUILD_BENCHMARK=on .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    popd
}

function rocSOLVER() { 
    f1 rocSOLVER 
}
function hipBLAS() { 
    f1 hipBLAS 
}
function hipSPARSE() { 
    f1 hipSPARSE 
}
function rocSPARSE() { 
    f1 rocSPARSE 
}
function rocFFT() { 
    f1 rocFFT 
}

function HIP_Examples() {
    CURR_BUILD=HIP-Examples
    i=$CURR_BUILD
    build_entry $i
    pushd $ROCM_SRC_FOLDER/$i
    ./test_all.sh | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    popd

}

function MIVisionX() {
    CURR_BUILD=MIVisionX
    build_entry $CURR_BUILD
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir build; cd build
    #python MIVisionX-setup.py
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    cmake .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    popd
}

function rocRAND() {
    f5 rocRAND
}

function rccl() {
    f5 rccl
}
function f5() {
    CURR_BUILD=$1
    build_entry $CURR_BUILD
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    # this no longer working.

    if [[ $CURR_BUILD == "rocRAND" ]] ; then
        ./install -idt $INSTALL_SH_PACKAGE 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    else
        ./install.sh -idt $INSTALL_SH_PACKAGE 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    fi
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    # fixed install.sh by adding chrpath in yum. keep for a while and delete afterward.
    #mkdir build ; cd build 
    #CXX=/opt/rocm/bin/hipcc cmake .. 2>&1 | tee -a $CURR_BUILD
    #make -j`nproc` install 2>&1 | tee -a $CURR_BUILD
    #popd
}

function rocALUTION () {
    i=rocALUTION
    CURR_BUILD=$i
    build_entry $i
    pushd $ROCM_SRC_FOLDER/$i
    mkdir build ; cd build

    cmake .. -DROCM_PATH=$ROCM_INST_FOLDER -DSUPPORT_HIP=ON | tee $LOG_DIR/$CURR_BUILD-1.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 1" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 2" >> $LOG_SUMMARY ; fi
    make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-3.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 3" >> $LOG_SUMMARY ; fi
    popd

}

function ROCm_OpenCL_Runtime() {

    pushd $ROCM_SRC_FOLDER/ROCclr
    PWD=$ROCM_SRC_FOLDER/ROCclr/build
    OPENCL_DIR=$ROCM_SRC_FOLDER/ROCm-OpenCL-Runtime/
    ROCclr_DIR=$ROCM_SRC_FOLDER/ROCclr/
    OLDPWD=$ROCM_SRC_FOLDER/ROCclr

    #CURR_BUILD=ROCclr
    #build_entry $CURR_BUILD
    #mkdir build; cd build
    #cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr .. 2>&1 | tee $LOG_DIR/ROCclr-1.log
    #if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-1" >> $LOG_SUMMARY ; fi

    #make -j$NPROC 2>&1 | tee -$LOG_DIR/ROCclr-2.log
    #if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-2" >> $LOG_SUMMARY ; fi

    #make install 2>&1 | tee -a $LOG_DIR/ROCclr-3.log
    #popd

    CURR_BUILD=ROCm-OpenCL-Runtime
    build_entry $CURR_BUILD
    cd $ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir -p build; cd build
    cmake -DUSE_COMGR_LIBRARY=ON -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER//ROCclr/build;/opt/rocm/" ..  | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

}

function ROCmValidationSuite() {
    f4 ROCmValidationSuite
}

function f4 () {
    i=$1
    CURR_BUILD=$i
    pushd $ROCM_SRC_FOLDER/ROCmValidationSuite
    mkdir build ; cd build
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-1" >> $LOG_SUMMARY ; fi
    cmake ..  2>&1 | tee -a $LOG_DIR/$CURR_BUILD-1.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-2" >> $LOG_SUMMARY ; fi
    make -j`nproc` | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make install | tee -a $LOG_DIR/$CURR_BUILD-3.log
    popd
}

pushd $ROCM_SRC_FOLDER
$COMP
ret=$?
popd
exit $ret

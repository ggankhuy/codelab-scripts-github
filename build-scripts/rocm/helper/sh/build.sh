source sh/common.sh

for var in "$@"
do
    if [[ $var == *"comp="* ]]  ; then
        comp=`echo $var | cut -d '=' -f2`
        COMP=$comp
        echo COMP old: $COMP
        COMP=$(echo $COMP | sed "s/-/_/g")
        echo COMP new: $COMP
    fi
done

function llvm() {
    pushd $ROCM_SRC_FOLDER
    CURR_BUILD=llvm-project
    build_entry $CURR_BUILD
    pushd $CURR_BUILD
    mkdir build ; cd build
    echo "cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$VERSION.$MINOR_VERSION/llvm -DCMAKE_BUILD_TYPE=Release"
    sleep 10
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$VERSION.$MINOR_VERSION/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC 2>&1 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    popd
    popd
}

function rocminfo()  {
    pushd $ROCM_SRC_FOLDER
    CURR_BUILD=rocminfo
    build_entry $i
    pushd $CURR_BUILD
    mkdir build; cd build
    cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm/ .. 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    popd
}

function ROCm_Device_Lib() {
    setup_root_rocm_softlink
    pushd $ROCM_SRC_FOLDER
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
    pushd $ROCM_SRC_FOLDER
    i=ROCT-Thunk-Interface
    CURR_BUILD=$i
    build_entry $i
    pushd $ROCM_SRC_FOLDER/$i
    mkdir build; cd build
    cmake .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    popd
}


$COMP


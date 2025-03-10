echo "build.sh entered..."
#   Command line variables passed down from python script. Do not put any other declaration of variables here.

CONFIG_BUILD_LLVM=1
CONFIG_BUILD_CMAKE=0
CONFIG_BUILD_PY=0
CONFIG_BUILD_FAST=0
CONFIG_BUILD_PACKAGE=0
CONFIG_TEST_MODE=0
CONFIG_INSTALL_PATH=""
CONFIG_BYPASS_PACKAGES_INSTALL=0
CONFIG_CLEAN_BUILD=1
BUILD_RESULT_PASS=0
BUILD_RESULT_FAIL=1
BUILD_RESULT_UNKNOWN=2

CONFIG_BUILD_TARGET_GPU_ONLY=1
TARGET_GFX=""
TARGET_GFX_OPTION1=""
TARGET_GFX_OPTION2=""
TARGET_GFX_OPTION3=""
CONFIG_GFX_OVERRIDE=""

for var in "$@"
do
    echo var: $var
    case "$var" in
        *comp=*)
            comp=`echo $var | cut -d '=' -f2`
            COMP=$comp
            COMP_OLD=$comp
            echo COMP old: $COMP

            # does not work well with rocm-cmake folder.
            if [[ $COMP != "rocm-cmake" ]] ; then
                COMP=$(echo $COMP | sed "s/-/_/g")
            fi
            echo COMP new: $COMP
            ;;

        *verminor=*)
            echo "processing var: $var"
            verminor=`echo $var | awk -F'=' '{print $2}'`
            ;;

        *vermajor=*)
            echo "processing var: $var"
            vermajor=`echo $var | awk -F'=' '{print $2}'`
            ;;

        *--path=*)
            echo "processing var: $var"
            CONFIG_INSTALL_PATH=`echo $var | cut -d '=' -f2`
            ;;

        *--llvmno*)
            echo "Will bypass llvm build."
            CONFIG_BUILD_LLVM=0
            ;;

        *--cmakeno*)
            echo "Will bypass cmake build."
            CONFIG_BUILD_CMAKE=0
            ;;

        *--gfx*)
            echo "Will bypass cmake build."
            CONFIG_GFX_OVERRIDE=`echo $var | cut -d '=' -f2`
            ;;

        *--cmake*)
            echo "Will force cmake build."
            CONFIG_BUILD_CMAKE=1
            ;;

        *--pyno*)
            echo "Will bypass python build."
            CONFIG_BUILD_PY=0
            ;;

        *--fast*)
            echo "Will speed up build whenever possible."
            CONFIG_BUILD_FAST=1
            ;;

        *--package*)
            echo "Will create package whenever possible."
            CONFIG_BUILD_PACKAGE=1
            ;;
        *--pkgno*)
            echo "Will create package whenever possible."
            CONFIG_BYPASS_PACKAGES_INSTALL=1
            ;;
        *--testmode*)
            echo "Will perform test mode only."
            CONFIG_TEST_MODE=`echo $var | cut -d '=' -f2`
            ;;
        *)
            #echo "Unknown cmdline parameter: $var" ; exit 1
            echo "Warning: Unknown cmdline parameter: $var"
    esac
        
done

source sh/common.sh 
source sh/prebuild.sh

if [[ $CONFIG_BUILD_TARGET_GPU_ONLY -eq 1 ]] ; then
     TARGET_GFX=`sudo rocminfo | grep gfx | head -1 | awk {'print $2'}`
    [[ -z $CONFIG_GFX_OVERRIDE ]] || TARGET_GFX=$COFNIG_GFX_OVERRIDE
    [[ -z $TARGET_GFX ]] || TARGET_GFX_OPTION1=" -a $TARGET_GFX"
    [[ -z $TARGET_GFX ]] || TARGET_GFX_OPTION2=" -D AMDGPU_TARGETS=$TARGET_GFX"
    [[ -z $TARGET_GFX ]] || TARGET_GFX_OPTION3=" -D GPU_TARGETS=$TARGET_GFX"
fi


ERROR_CODE=$?
if [[ $ERROR_CODE -ne 0 ]] ; then 
    echo "Error during prebuild stage: error code: $ERROR_CODE" ;
    exit $ERROR_CODE
else
    echo "Prebuild stage is OK. continuing."
fi

echo build.sh: PATH: $PATH

set_os_type

if [[ CONFIG_BYPASS_PACKAGES_INSTALL -eq 1 ]] ; then
    echo "Bypass installing packages..."
else
    echo "Installing packages..."
    echo "PKG_EXEC: $PKG_EXEC"
    case "$PKG_EXEC" in
       "apt")
            # vim-common: rocm5.5 rocr-runtime.
            # libnuma-dev: rocm5.5 roct-thunk-interface.
            install_packages cmake chrpath libpci-dev libstdc++-12-dev cmake make half vim-common libnuma-dev pkg-config rpm 
          ;;
       "yum")
            # rocprof: rocm-llvm-devel, libdwarf-devel (not sure if this is needed).
            # rocSolver: fmt-devel
            # rocblas: python3-joblib
            # MIOpenGEMM: yaml-cpp
            install_packages cmake libstdc++-devel libpci-devel gcc g++ elfutils-libelf-devel numactl-devel libdrm-devel pciutils-devel vim-common libX11-devel mesa-libGL-devel libdwarf-devel rocm-llvm-devel fmt-devel yaml-cpp dpkg
          ;;
       "yum")
            install_packages cmake 
          ;;
       *)
            echo "Unable to determine PKG_EXEC or unsupport/unknown package installer: $PKG_EXEC. Installing linux packages are skipped."
        ;;    
    esac
    # hipBlastLT: joblib
    # rocprofiler: lxml barectf LibDw
    install_pip_libs CppHeaderParser joblib lxml barectf LibDw
fi

CONFIG_INSTALL_PREFIX="/opt/rocm"

if [[ ! -z $CONFIG_INSTALL_PATH ]] ; then
    CONFIG_INSTALL_PREFIX="$CONFIG_INSTALL_PATH"
    echo "Installing clr into non-standard location: $CONFIG_INSTALL_PREFIX..."
    sleep 1
fi

# temporarily removing -i option as install.sh does not support allowerasing and -i causes allowerasing 
# to be needed.

# p1 - rocm component
# p2 - cmake or install or install.sh
# p3 and on
#   params=parameters for p2.
#   gfx=target gpu
#   target=build target for make.

# cmake 

function f0() {
    counter=0
    PARAMS=""; BUILD_TARGET=""; GFX=""
    for var in "$@"
    do
        echo var: $var, counter: $counter
        if [[ $counter -eq 0 ]] ; then CURR_BUILD=$var ; counter=$((counter+1)) ; continue ; fi
        if [[ $counter -eq 1 ]] ; then INSTALL_OPTION=$var ; counter=$((counter+1)) ; continue ; fi

        case "$var" in
            params=*)
                PARAMS=`echo $var | cut -d '=' -f2-`
                ;;
            target=*)
                BUILD_TARGET=`echo $var | cut -d '=' -f2`
                ;;
            gfx=*)
                GFX=`echo $var | cut -d '=' -f2`
                ;;
            *)
               [[ $counter == 1 ]] || [[ $counter == 0 ]] || echo "Warning: Unknown cmdline parameter: $var"
        esac
        counter=$((counter+1))
    done

    build_entry $CURR_BUILD
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS
    case "$INSTALL_OPTION" in
        cmake)
            [[ -z $CONFIG_CLEAN_BUILD ]] || rm -rf build
            mkdir build ; pushd build
            pwd
            echo HIP_CXX_COMPILER=hipcc cmake $GFX $PARAMS .. | tee $LOG_DIR/$CURR_BUILD.log
            HIP_CXX_COMPILER=hipcc cmake $GFX $PARAMS .. | tee $LOG_DIR/$CURR_BUILD.log
            if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
            make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
            if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
            make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
            if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
            popd
            ;;        
        install*)
            INSTALLER=$INSTALL_OPTION
            if [[ -f ./$INSTALLER ]] ; then
                [[ -z $CONFIG_CLEAN_BUILD ]] || rm -rf build
                 ./$INSTALLER $GFX $PARAMS 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
            else
                echo "Error: Installer doesnot exist: INSTALLER: $INSTALLER"
                BUILD_RESULT=$BUILD_RESULT_FAIL
            fi
            ;;
        *)
            echo "Warning : Unknown install option: INSTALL_OPTION: $INSTALL_OPTION"
    esac
    build_exit $CURR_BUILD $BUILD_RESULT
}

function f1() {
    i=$1
    CURR_BUILD=$i
    build_entry $i
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$i
    ./install.sh $TARGET_GFX_OPTION1 -cd 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    #./install.sh -icd 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function f5() {
    CURR_BUILD=$1
    build_entry $CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS
    P2=$2

    [[ -z $P2 ]] || INSTALL_PARAMS="-icd"  && INSTALL_PARAMS=$P2

    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    # this no longer working.

    CONFIG_INSTALL_PREFIX="--prefix=/opt/rocm"
    [[ $CURR_BUILD == "rocPRIM" ]] && CONFIG_INSTALL_PREFIX=""

    if [[ $CURR_BUILD == "rocRAND" ]] || [[ $CURR_BUILD == "rocPRIM" ]] ; then
        ./install $INSTALL_PARAMS $CONFIG_INSTALL_PREFIX  2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    else
        ./install.sh $TARGET_GFX_OPTION1 $INSTALL_PARAMS $CONFIG_INSTALL_PREFIX  2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    fi
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi

    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}



# build helper functions
# cmake.. + make is used
function f2() {
    i=$1
    CURR_BUILD=$i
    build_entry $i
    INSTALL_TARGET
    pushd $ROCM_SRC_FOLDER/$i
    mkdir build; cd build
    BUILD_RESULT=$BUILD_RESULT_PASS

    HIP_CXX_COMPILER=hipcc cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_BENCHMARK=on .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function f2a() {
    i=$1
    CURR_BUILD=$i
    build_entry $i
    pushd $ROCM_SRC_FOLDER/$i
    mkdir build; cd build
    BUILD_RESULT=$BUILD_RESULT_PASS

    rm -rf ./*
    CXX=hipcc cmake .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function f3 () {
    i=$1
    build_entry $i
    CURR_BUILD=$i
    BUILD_TARGET=package
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $COMP_OLD
    mkdir build; cd build
    cmake -DCMAKE_PREFIX_PATH=$CONFIG_INSTALL_PREFIX -DCMAKE_INSTALL_PREFIX=$CONFIG_INSTALL_PREFIX .. 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log

}
function f4 () {
    i=$1
    CURR_BUILD=$i
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$i
    mkdir build ; cd build
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-1" >> $LOG_SUMMARY ; fi
    cmake ..  2>&1 | tee -a $LOG_DIR/$CURR_BUILD-1.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-2" >> $LOG_SUMMARY ; fi
    make -j`nproc` | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make install | tee -a $LOG_DIR/$CURR_BUILD-3.log
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function f6() {
    i=$1
    CURR_BUILD=$i
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$i
    pip3 install -r requirements.txt 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-1.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-1" >> $LOG_SUMMARY ; fi
    ./build.sh 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail-2" >> $LOG_SUMMARY ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

# 1) ./install.sh -idt is used
# p1 = rocm component to build
# p2 = install parameters (install or install.sh)
# 

function llvm() {
    CONFIG_INSTALL_PREFIX_LLVM=/opt/rocm-$VERSION_MAJOR.$VERSION_MINOR/llvm
    if [[ ! -z $CONFIG_INSTALL_PATH ]] ; then
        echo "Installing llvm into non-standard location: $CONFIG_INSTALL_PATH..."
        sleep 10
        CONFIG_INSTALL_PREFIX_LLVM=$CONFIG_INSTALL_PATH/llvm
    fi

    f0 llvm-project "cmake" \
    "params= \
        -G \"Unix Makefiles\" \
        -DCMAKE_INSTALL_PREFIX=$CONFIG_INSTALL_PREFIX_LLVM \
        -DCMAKE_BUILD_TYPE=Release"
        -DLLVM_ENABLE_PROJECTS=\"clang;lld;lldb;clang-tools-extra;compiler-rt\"
}

# phasing out in favor of f0, once works, get rid of it.
function llvm_0() {    
    CURR_BUILD=llvm-project
    build_entry $CURR_BUILD
    pushd $CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS

    mkdir build ; cd build
    CONFIG_INSTALL_PREFIX_LLVM=/opt/rocm-$VERSION_MAJOR.$VERSION_MINOR/llvm
    if [[ ! -z $CONFIG_INSTALL_PATH ]] ; then
        echo "Installing llvm into non-standard location: $CONFIG_INSTALL_PATH..."
        sleep 10
        CONFIG_INSTALL_PREFIX_LLVM=$CONFIG_INSTALL_PATH/llvm
    fi
    echo "cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$CONFIG_INSTALL_PREFIX_LLVM -DCMAKE_BUILD_TYPE=Release"
    cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=$CONFIG_INSTALL_PREFIX_LLVM -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC 2>&1 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

# cmake .. + make

function rocminfo()  {
    f0 rocminfo cmake "params=-DCMAKE_PREFIX_PATH=$CONFIG_INSTALL_PREFIX -DCMAKE_INSTALL_PREFIX=$CONFIG_INSTALL_PREFIX"
#   f3 rocminfo
}

function MIOpenGEMM() {
    f0 MIOpenGEMM cmake
#   f3 MIOpenGEMM
}

# obsolete, here only in case needed. 

function ROCm_Device_Lib() {
    setup_root_rocm_softlink
    CURR_BUILD=ROCm-Device-Libs
    build_entry $CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS
    DEVICE_LIBS=$ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir -p "$DEVICE_LIBS/build"
    cd "$DEVICE_LIBS/build"
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" -DCMAKE_INSTALL_PREFIX=/opt/rocm/ .. 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    #cp *deb *rpm $CONFIG_BUILD_PKGS_LOC/
    build_exit $CURR_BUILD $BUILD_RESULT
}

function composable_kernel() {
    f0 composable_kernel cmake \
        gfx=$TARGET_GFX_OPTION3 \
        params=" -D CMAKE_PREFIX_PATH=/opt/rocm -D CMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc -D BUILD_TYPE=Release"
}

function rocm-cmake() {
    f0 rocm-cmake cmake 
}
function ROCmValidationSuite() {
    f0 ROCmValidationSuite cmake
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
    f6 rocprofiler
}

function rocr_debug_agent() {
    f3 rocprofiler
}

function clang_ocl() {
    f3 clang_ocl
}

function COMGR() {

    CURR_BUILD=ROCm-CompilerSupport
    build_entry $CURR_BUILD
    LLVM_PROJECT=$ROCM_SRC_FOLDER/llvm-project
    DEVICE_LIBS=$ROCM_SRC_FOLDER/ROCm-Device-Libs
    COMGR=$ROCM_SRC_FOLDER/$CURR_BUILD/lib/comgr

    BUILD_RESULT=$BUILD_RESULT_PASS

    mkdir -p "$DEVICE_LIBS/build"
    pushd "$DEVICE_LIBS/build"
    pwd

    cmake -DCMAKE_PREFIX_PATH=$CONFIG_INSTALL_PREFIX -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" .. | tee  -a $LOG_DIR/$CURR_BUILD.log
    make -j$NPROC $BUILD_TARGET  2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd

    mkdir -p $ROCM_SRC_FOLDER/$CURR_BUILD/lib/comgr/build
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    cd lib/comgr/build
    pwd
    echo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" .. | tee  -a $LOG_DIR/$CURR_BUILD.log
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" .. | tee  -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET  2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    #make test 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    #if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function protobuf() {
    CURR_BUILD=protobuf
    build_entry $CURR_BUILD

    git clone https://github.com/protocolbuffers/protobuf.git
    cd $CURR_BUILD 
    git checkout v3.16.0
    git submodule update --init --recursive
    mkdir build ; cd build

    BUILD_RESULT=$BUILD_RESULT_PASS
    cmake ../cmake -Dprotobuf_BUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_SYSCONFDIR=/etc -DCMAKE_POSITION_INDEPENDENT_CODE=ON -Dprotobuf_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi

    make -j`nproc` 
    make $INSTALL_TARGET    
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    cd ../..
    build_exit $CURR_BUILD $BUILD_RESULT
}

function AMDMIGraphX() {
    #./tools/install_prereqs.sh

    CURR_BUILD=AMDMIGraphX
    BUILD_RESULT=$BUILD_RESULT_PASS

    # following commented lines replaced by install_prereqs.sh, hopefully, intest.
    pip3 install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz | tee  $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi

    BUILD_RESULT=$BUILD_RESULT_PASS
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    rbuild build -d depend --cxx=/opt/rocm/llvm/bin/clang++ | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    mkdir build; cd build
    CXX=/opt/rocm/llvm/bin/clang++ cmake .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function rocBLAS() {

    # checkout tensile 

    tensileTag=`cat $ROCM_SRC_FOLDER/$i/tensile_tag.txt`
    [[ ! -z $tensile_tag ]] || BUILD_RESULT=$BUILD_RESULT_FAIL
    return $BUILD_RESULT_FAIL

    if [[ $tensileTag ]] ; then
        [[ ! -z $CONFIG_CLEAN_BUILD ]] || rm -rf Tensile
        git clone https://github.com/ROCmSoftwarePlatform/Tensile.git
        pushd Tensile
        git checkout $tensileTag
        tensileTagOK=`git log | grep $tensileTag`
        if [[ -z $tensileTagOK ]] ; then
            echo "Warning: unable to checkout Tensile with commit tag: $tensileTag, will do a full build"
        else 
            FAST_BUILD_ROCBLAS_OPT=$FAST_BUILD_ROCBLAS_OPT" -t $ROCM_SRC_FOLDER/Tensile "
        fi
        popd
    fi
    sed -i 's/\"python3-joblib\"//g' ./install.sh

    f0 install.sh gfx=$TARGET_GFX_OPTION params="$FAST_BUILD_ROCBLAS_OPT -cd"
    #./install.sh $TARGET_GFX_OPTION $FAST_BUILD_ROCBLAS_OPT | tee $LOG_DIR/$CURR_BUILD.log
    #if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    #popd
    
    #$PKG_EXEC install -y $ROCM_SRC_FOLDER/rocBLAS/build/release/*.$PKG_EXT 2>&1 | tee -a $CURR_BUILD.log
    #build_exit $CURR_BUILD $BUILD_RESULT
}

function MIOpen() {
    i=MIOpen
    CURR_BUILD=$i
    build_entry $i
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $i
    cmake -P install_deps.cmake --minimum | tee $LOG_DIR/$CURR_BUILD-1.log
    mkdir build; cd build
    rm -rf ./*
    echo "current dir: "  | tee $LOG_DIR/$CURR_BUILD-2.log
    pwd 2>&1 | tee $LOG_DIR/$CURR_BUILD-2.log
    CXX=/opt/rocm/llvm/bin/clang++ cmake $CONFIG_MIOPEN_BUILD_ROCBLAS -DMIOPEN_BACKEND=HIP -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ .. 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 1." >> $LOG_SUMMARY ; fi

    echo "make -j$NPROC $BUILD_TARGET 2>&1" | tee $LOG_DIR/$CURR_BUILD-3.log
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-3.log

    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 2." >> $LOG_SUMMARY ; fi
    echo "make $INSTALL_TARGET 2>&1" | tee $LOG_DIR/$CURR_BUILD-4.log

    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-4.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 3." >> $LOG_SUMMARY ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function hipAMD() {
    CURR_BUILD=hipamd
    build_entry $CURR_BUILD
    cd $ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir build ; cd build
    BUILD_RESULT=$BUILD_RESULT_PASS

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
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    make -j$NPROC 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    #make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    build_exit $CURR_BUILD $BUILD_RESULT
}

function clr() {
    HIP_FOLDER=$ROCM_SRC_FOLDER/HIP
    f0 clr cmake  "params=-DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_FOLDER"
}
function clr_0() {
    CURR_BUILD=clr
    build_entry $CURR_BUILD
    PWD=`pwd`
    BUILD_RESULT=$BUILD_RESULT_PASS

    cd $ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir build
    pushd build
    if [[ ! -d $HIP_FOLDER ]] ; then echo "Fail: Unable to find DHIP_COMMON_DIR: $HIP_FOLDER:" >> $LOG_SUMMARY ; fi
    cmake .. -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_FOLDER
    make -j$NPROC 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    #make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    make -j$NPROC 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}
function clr_old() {
    CURR_BUILD=clr
    build_entry $CURR_BUILD
    cd $ROCM_SRC_FOLDER/$CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS

    mkdir build ; cd build

    pushd ../..
    export HIPAMD_DIR="$(readlink -f hipamd)"
    export HIP_DIR="$(readlink -f hip)"
    export ROCCLR_DIR="$(readlink -f ROCclr)"
    export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime)"
    echo HIPAMD_DIR: $HIPAMD_DIR, HIP_DIR: $HIP_DIR, ROCclr_DIR: $ROCclr_DIR, OPENCL_DIR: $OPENCL_DIR
    popd

    HIP_CLANG_PATH=$CONFIG_INSTALL_PREFIX/llvm/bin CXX=$CONFIG_INSTALL_PREFIX/llvm/bin/clang++ cmake .. \
        -DCMAKE_CXX_COMPILER=$CONFIG_INSTALL_PREFIX/llvm/bin/clang++ \
        -DCMAKE_PREFIX_PATH=$CONFIG_INSTALL_PREFIX \
        -DClang_DIR=$CONFIG_INSTALL_PREFIX/llvm/lib/cmake/clang/ \
        -DCLR_BUILD_HIP=ON \
        -DHIP_COMMON_DIR=$ROCM_SRC_FOLDER/HIP \
        -DROCCLR_PATH=$ROCM_SRC_FOLDER/clr/rocclr \
        -DHIPCC_BIN_DIR=$ROCM_SRC_FOLDER/HIPCC/bin 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    make -j`nproc`  | tee -a $LOG_DIR/$CURR_BUILD.log
    build_exit $CURR_BUILD $BUILD_RESULT 
}

function rocPRIM() {
    f5 rocPRIM "-icp"
}
function hipCUB() {
    f2 hipCUB
}

function rocThrust() {
    f2a rocThrust
}

function MIVisionX() {
    f2a MIVisionX
}
function rocSOLVER() { 
    f1 rocSOLVER 
}
function hipBLAS() { 
    f1 hipBLAS 
}
function hipBLASLt() { 
    f1 hipBLASLt 
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

#-DCMAKE_PREFIX_PATH

function HIP_Examples() {
    CURR_BUILD=HIP-Examples
    i=$CURR_BUILD
    build_entry $i
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$i
    ./test_all.sh | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function MIVisionX() {
    CURR_BUILD=MIVisionX
    build_entry $CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$CURR_BUILD

    mkdir build; cd build
    #python MIVisionX-setup.py
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    cmake .. | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function rocRAND() {
    CURR_BUILD=rocRAND
    build_entry $CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir build; cd build

    #pushd ../..
    #export HIPAMD_DIR="$(readlink -f hipamd)"
    #export HIP_DIR="$(readlink -f hip)"
    #echo HIPAMD_DIR: $HIPAMD_DIR, HIP_DIR: $HIP_DIR, ROCclr_DIR: $ROCclr_DIR, OPENCL_DIR: $OPENCL_DIR
    #popd
    #sudo ln -s $ROCM_SRC_FOLDER/HIP $ROCM_SRC_FOLDER/hip
    #cmake -DHIP_COMMON_DIR=$HIP_DIR -DAMD_OPENCL_PATH=$OPENCL_DIR -DROCCLR_PATH=$ROCCLR_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" .. 2>&1 | tee $LOG$

    hip_DIR="$ROCM_SRC_FOLDER/hipamd"
    #CXX=hipcc cmake -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER/hipAMD/" -DBUILD_BENCHMARK=ON .. 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    CXX=hipcc cmake $TARGET_GFX_OPTION2 -DBUILD_HIPRAND=OFF -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER/hipAMD/" -DBUILD_BENCHMARK=ON .. 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    make -j`nproc` install | tee $LOG_DIR/$CURR_BUILD.log
#   f5 rocRAND
    build_exit $CURR_BUILD $BUILD_RESULT
}

function rccl() {
    f0 rccl "install.sh" "params=-dpt" "gfx=-l"
}

function rocALUTION () {
    i=rocALUTION
    CURR_BUILD=$i
    build_entry $i
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$i
    mkdir build ; cd build

    cmake .. -DROCM_PATH=$ROCM_INST_FOLDER -DSUPPORT_HIP=ON | tee $LOG_DIR/$CURR_BUILD-1.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 1" >> $LOG_SUMMARY ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-2.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 2" >> $LOG_SUMMARY ; fi
    make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD-3.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail 3" >> $LOG_SUMMARY ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function ROCm_OpenCL_Runtime() {
    pushd $ROCM_SRC_FOLDER/ROCclr
    PWD=$ROCM_SRC_FOLDER/ROCclr/build
    OPENCL_DIR=$ROCM_SRC_FOLDER/ROCm-OpenCL-Runtime/
    ROCclr_DIR=$ROCM_SRC_FOLDER/ROCclr/
    OLDPWD=$ROCM_SRC_FOLDER/ROCclr
    BUILD_RESULT=$BUILD_RESULT_PASS

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
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    build_exit $CURR_BUILD $BUILD_RESULT
}

function ROCmValidationSuite() {
    f4 ROCmValidationSuite
}

function ROCR_Runtime() {
    i=ROCR-Runtime
    CURR_BUILD=$i
    build_entry $i
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$i/src
    mkdir build; cd build
    cmake -DCMAKE_PREFIX_PATH=$CONFIG_INSTALL_PREFIX -DIMAGE_SUPPORT=OFF .. 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function roctracer() {
    i=roctracer
    CURR_BUILD=$i
    build_entry $i
    BUILD_RESULT=$BUILD_RESULT_PASS

    $PKG_EXEC install rpm -y
    pushd $ROCM_SRC_FOLDER/$i
    ./build.sh
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function ROCgdb() {
    CURR_BUILD=ROCgdb
    build_entry $CURR_BUILD
    BUILD_RESULT=$BUILD_RESULT_PASS

    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    ./configure
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$NPROC
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    build_exit $CURR_BUILD $BUILD_RESULT
}

function hip_tests() {
    #not sure if this export statement is needed. at the time of this writing
    # develop branch builds ok on mi300, 6.2.x not, mi250 builds ok.
    #export HIP_TESTS_DIR="$(readlink -f hip-tests)"
    CURR_BUILD=hip-tests
    BUILD_RESULT=$BUILD_RESULT_PASS

    build_entry $CURR_BUILD
    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    mkdir -p build; cd build
    cmake ../catch/  -DHIP_PLATFORM=amd
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    make -j$(nproc) build_tests
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    popd
    if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; BUILD_RESULT=$BUILD_RESULT_FAIL ; fi
    build_exit $CURR_BUILD $BUILD_RESULT
}
if [[ $CONFIG_BUILD_PY -eq 1 ]] ; then
    echo "Building and installing python..."
    install_python
else
    echo "Bypassing python..."
fi

if [[ $CONFIG_BUILD_CMAKE -eq 1 ]] ; then
    echo "Building and installing cmake.../not implemented/"
    #install_cmake
else
    echo "Bypassing cmake..."
fi

pushd $ROCM_SRC_FOLDER
if [[ $CONFIG_BUILD_LLVM -eq 1 ]] ; then
    if [[ $CONFIG_TEST_MODE -eq 1 ]] ; then
        echo "build.sh: TEST_MODE: building llvm..."
    fi
    llvm

fi
if [[ $CONFIG_TEST_MODE -eq 1 ]] ;  then
    echo "build.sh: TEST_MODE: building $COMP..."
    exit 0
else
    echo "build.sh: building $COMP..."
    $COMP
    ret=$?
    exit $ret
fi
popd


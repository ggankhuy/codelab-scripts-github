
p1=$1
if [[ -z $p1 ]] ; then
	echo "You need to specify rocm version."
	exit 1
fi
pushd llvm-project
LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
ROCM_SRC_FOLDER=~/ROCm-$p1
ROCM_INST_FOLDER=/opt/rocm-$p1-0/
function setup_root_rocm_softlink () {
	rm ~/ROCm
	ln -s $ROCM_SRC_FOLDER  ~/ROCm
	if [[ $? -ne  0 ]] ; then 
		echo "Error during setting up the softlink ~/ROCm"
		ls -l ~
		exit 1
	fi
}

function setup_opt_rocm_softlink () {
	rm /opt/rocm
	ln -s /opt/$ROCM_INST_FOLDER/ /opt/rocm

	if [[ $? -ne  0 ]] ; then 
		echo "Error during setting up the softlink /opt/rocm"
		ls -l /opt
		exit 1
	fi
}

setup_root_rocm_softlink
setup_opt_rocm_softlink

pushd $ROCM_SRC_FOLDER/ROCm-CompilerSupport
cd lib/comgr/
LLVM_PROJECT=$ROCM_SRC_FOLDER/llvm-project
DEVICE_LIBS=$ROCM_SRC_FOLDER/ROCm-Device-Libs/
COMGR=$ROCM_SRC_FOLDER/ROCm-CompilerSupport/lib/comgr

setup_root_rocm_softlink
mkdir -p "$DEVICE_LIBS/build"
cd "$DEVICE_LIBS/build"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" .. | tee $LOG_DIR/ROCm-CompilerSupport.log
make -j`nproc` ; make install | tee -a $LOG_DIR/ROCm-CompilerSupport.log

mkdir -p "$COMGR/build"
cd "$COMGR/build"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" .. | tee  $LOG_DIR/ROCm-CompilerSupport.log
make -j$NPROC  | tee -a $LOG_DIR/ROCm-CompilerSupport.log
make test | tee -a $LOG_DIR/ROCm-CompilerSupport.log
make install | tee -a $LOG_DIR/ROCm-CompilerSupport.log
popd

pushd ROCclr
PWD=/root/ROCm/ROCclr/build
OPENCL_DIR=/root/ROCm/ROCm-OpenCL-Runtime/
ROCclr_DIR=/root/ROCm/ROCclr/
OLDPWD=/root/ROCm/ROCclr

mkdir build; cd build
cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr .. | tee $LOG_DIR/ROCclr.log
make -j$NPROC install | tee -a $LOG_DIR/ROCclr.log
popd


exit 0

echo  qualified codes
mkdir -p $LOG_DIR
mkdir build ; cd build
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$p1.0/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm | tee $LOG_DIR/llvm.log
make -j$NPROC  | tee -a $LOG_DIR/llvm.log
make install  | tee -a $LOG_DIR/llvm.log
popd

pushd rocBLAS
./install.sh -icd | tee $LOG_DIR/rocBLAS.log
popd



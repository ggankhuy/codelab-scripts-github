
p1=$1
if [[ -z $p1 ]] ; then
	echo "You need to specify rocm version."
	exit 1
fi
pushd llvm-project
LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
rm ~/ROCm
ln -s ~/ROCm-$p1.0 ~/ROCm

function setup_root_rocm_softlink () {
	if [[ $? -ne  0 ]] ; then 
		echo "Error during setting up the softlink ~/ROCm"
		ls -l ~
		exit 1
	fi
}

function setup_opt_rocm_softlink () {
	rm /opt/rocm
	ln -s /opt/rocm-$p1-0/ /opt/rocm

	if [[ $? -ne  0 ]] ; then 
		echo "Error during setting up the softlink /opt/rocm"
		ls -l /opt
		exit 1
	fi
}

setup_root_rocm_softlink
setup_opt_rocm_softlink

mkdir -p $LOG_DIR
mkdir build ; cd build
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$p1.0/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm | tee $LOG_DIR/llvm.log
make -j$NPROC  | tee -a $LOG_DIR/llvm.log
make install  | tee -a $LOG_DIR/llvm.log
popd

pushd rocBLAS
./install.sh -icd | tee $LOG_DIR/rocBLAS.log
popd


pushd ROCm-CompilerSupport
cd lib/comgr/
LLVM_PROJECT=~/ROCm/llvm-project
DEVICE_LIBS=~/ROCm/ROCm-Device-Libs/
COMGR=~/ROCm/ROCm-CompilerSupport/lib/comgr

mkdir -p "$DEVICE_LIBS/build"
cd "$DEVICE_LIBS/build"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" .. | tee $LOG_DIR/ROCm-CompilerSupport.log
make -j`nproc` ; make install | tee -a $LOG_DIR/ROCm-CompilerSupport.log

mkdir -p "$COMGR/build"
cd "$COMGR/build"
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" .. | tee -a $LOG_DIR/ROCm-CompilerSupport.log
make -j$NPROC  | tee -a $LOG_DIR/ROCm-CompilerSupport.log
make test | tee -a $LOG_DIR/ROCm-CompilerSupport.log
make install | tee -a $LOG_DIR/ROCm-CompilerSupport.log
popd

pushd ROCclr
PWD=/root/ROCm/ROCclr/build
OPENCL_DIR=/root/ROCm/ROCm-OpenCL-Runtime/
ROCclr_DIR=/root/ROCm/ROCclr/
OLDPWD=/root/ROCm/ROCclr

setup_root_rocm_softlink
mkdir build; cd build
cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr .. | tee $LOG_DIR/ROCclr.log
make -j$NPROC install | tee -a $LOG_DIR/ROCclr.log
popd



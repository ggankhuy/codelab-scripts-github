
p1=$1
if [[ -z $p1 ]] ; then
	echo "You need to specify rocm version."
	exit 1
fi
export PATH=$PATH:/opt/rocm-4.2.0/llvm/bin/
LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
ROCM_SRC_FOLDER=~/ROCm-$p1
ROCM_INST_FOLDER=/opt/rocm-$p1.0/

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
	ln -s $ROCM_INST_FOLDER/ /opt/rocm

	if [[ $? -ne  0 ]] ; then 
		echo "Error during setting up the softlink /opt/rocm"
		ls -l /opt
		exit 1
	fi
}

mkdir -p $LOG_DIR
setup_root_rocm_softlink
setup_opt_rocm_softlink

pushd llvm-project
mkdir build ; cd build
cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$p1.0/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm | tee $LOG_DIR/llvm.log
make -j$NPROC  | tee -a $LOG_DIR/llvm.log
make install  | tee -a $LOG_DIR/llvm.log
popd

pushd rocBLAS
./install.sh -icd | tee $LOG_DIR/rocBLAS.log
popd

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

pushd $ROCM_SRC_FOLDER/ROCclr
PWD=$ROCM_SRC_FOLDER/ROCclr/build
OPENCL_DIR=$ROCM_SRC_FOLDER/ROCm-OpenCL-Runtime/
ROCclr_DIR=$ROCM_SRC_FOLDER/ROCclr/
OLDPWD=$ROCM_SRC_FOLDER/ROCclr

mkdir build; cd build
cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr .. | tee $LOG_DIR/ROCclr.log
make -j$NPROC install | tee -a $LOG_DIR/ROCclr.log
popd

cd $ROCM_SRC_FOLDER/HIP
mkdir build ; cd build
cmake -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER/ROCclr/build;/opt/rocm/" .. | tee $LOG_DIR/hip.log
make -j$NPROC | tee -a $LOG_DIR/hip.log
make install | tee -a $LOG_DIR/hip.log


pushd $ROCM_SRC_FOLDER/rccl
./install.sh -idt
popd

for i in rocm_smi_lib rocm_bandwidth_test rocminfo
do
	echo $building $i
	pushd $ROCM_SRC_FOLDER/$i
	mkdir build; cd build
	rm -rf ./*
	cmake .. | tee $LOG_DIR/$i.log
	make -j$NPROC | tee -a $LOG_DIR/$i.log
	make install | tee -a $LOG_DIR/$i.log
	popd
done

exit 0



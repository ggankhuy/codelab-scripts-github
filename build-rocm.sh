REPO_ONLY=0
NON_REPO_ONLY=0
p1=$1
CONFIG_TEST=0
for var in "$@"
do
    if [[ $var == "repo_only" ]]  ; then
        echo repo only specified: $var
        REPO_ONLY=1
    fi
    if [[ $var == "non_repo_only" ]]  ; then
        echo non repo only specified: $var
        NON_REPO_ONLY=1
    fi

    if [[ ! -z `echo "$var" | grep "test"` ]]  ; then
        echo "non-repo only specified: $var"
        CONFIG_TEST=1
    fi

    if [[ ! -z `echo "$var" | grep "ver="` ]]  ; then
        VERSION=`echo $var | cut -d '=' -f2`
    fi
done

if [[ $p1 == '--help' ]] || [[ $p1 == "" ]]   ; then
    echo "Usage: $0 <parameters>."
    echo "Parameters:"
    echo "repo_only - build only rocm repository components."
    echo "non_repo_only - build onlu non-rocm repository components"
    echo "ver=<rocm version> - specifying version is mandatory. i.e. 4.1, 4.2" 
    echo "test - test run only, will not build qualified builds."

    exit 0 ;
fi

if [[ -z $VERSION ]] ; then
	echo "You need to specify rocm version."
	exit 1
fi
export PATH=$PATH:/opt/rocm-4.2.0/llvm/bin/
LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
ROCM_SRC_FOLDER=~/ROCm-$VERSION
ROCM_INST_FOLDER=/opt/rocm-$VERSION.0/

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

echo "REPO_ONLY: $REPO_ONLY "
echo "NON_REPO_ONLY: $NON_REPO_ONLY"
sleep 5

if [[ $CONFIG_TEST == 0 ]] && [[ $REPO_ONLY == 1 ]] ; then
	CURR_BUILD=llvm-project
	pushd $CURR_BUILD
	mkdir build ; cd build
	cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$VERSION.0/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm | tee $LOG_DIR/$CURR_BUILD.log
	make -j$NPROC  | tee -a $LOG_DIR/$CURR_BUILD.log
	make install  | tee -a $LOG_DIR/$CURR_BUILD.log
	popd

	CURR_BUILD=ROCm-CompilerSupport
	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
	cd lib/comgr/
	LLVM_PROJECT=$ROCM_SRC_FOLDER/llvm-project
	DEVICE_LIBS=$ROCM_SRC_FOLDER/ROCm-Device-Libs/
	COMGR=$ROCM_SRC_FOLDER/$CURR_BUILD/lib/comgr

	setup_root_rocm_softlink

	mkdir -p "$DEVICE_LIBS/build"
	cd "$DEVICE_LIBS/build"
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" .. | tee $LOG_DIR/$CURR_BUILD.log
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
	make install | tee -a $LOG_DIR/$CURR_BUILD.log

	mkdir -p "$COMGR/build"
	cd "$COMGR/build"
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" .. | tee  -a $LOG_DIR/$CURR_BUILD.log
	make -j$NPROC  | tee -a $LOG_DIR/$CURR_BUILD.log
	make test | tee -a $LOG_DIR/$CURR_BUILD.log
	make install | tee -a $LOG_DIR/$CURR_BUILD.log
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

	CURR_BUILD=HIP
	cd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build ; cd build
	cmake -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER/ROCclr/build;/opt/rocm/" .. | tee $LOG_DIR/$CURR_BUILD.log
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
	make install | tee -a $LOG_DIR/$CURR_BUILD.log

	CURR_BUILD=ROCm-OpenCL-Runtime
	cd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir -p build; cd build
	cmake -DUSE_COMGR_LIBRARY=ON -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER//ROCclr/build;/opt/rocm/" ..  | tee $LOG_DIR/$CURR_BUILD.log
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
	make install | tee -a $LOG_DIR/$CURR_BUILD.log

	CURR_BUILD=rccl
	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
	./install.sh -idt | tee $LOG_DIR/$CURR_BUILD
	popd

	# for rocr_debug_agent!!

	apt install gcc g++ make cmake libelf-dev libdw-dev -y

	for i in rocm_smi_lib rocm_bandwidth_test rocminfo rocprofiler rocr_debug_agent MIOpenGEMM half clang-ocl rocm-cmake  ROCR-Runtime/src ROCT-Thunk-Interface
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		rm -rf ./*
		cmake .. | tee $LOG_DIR/$CURR_BUILD
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		make install | tee -a $LOG_DIR/$CURR_BUILD
		popd
	done

	for i in  rocThrust
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		rm -rf ./*
		CXX=/opt/rocm/hip/bin/hipcc cmake .. | tee $LOG_DIR/$CURR_BUILD
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		make install | tee -a $LOG_DIR/$CURR_BUILD
		popd
	done


	CURR_BUILD=ROCmValidationSuite
	pushd $ROCM_SRC_FOLDER/ROCmValidationSuite
	apt install libpciaccess-dev libpci-dev -y | tee  $LOG_DIR/$CURR_BUILD
	cmake ./ -B./build | tee -a $LOG_DIR/$CURR_BUILD
	make -C ./build | tee -a $LOG_DIR/$CURR_BUILD
	popd

	for i in rocPRIM hipCUB
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		rm -rf ./*
		CXX=/opt/rocm/hip/bin/hipcc cmake -DBUILD_BENCHMARK=on .. | tee $LOG_DIR/$CURR_BUILD
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		make install | tee -a $LOG_DIR/$CURR_BUILD
		popd
	done


	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build ; cd build
	CXX=/opt/rocm/hip/bin/hipcc cmake -DBUILD_BENCHMARK=ON ../. | tee $LOG_DIR/$CURR_BUILD
	make -j$NPROC install | tee -a $LOG_DIR/$CURR_BUILD
	popd

	# rocSPARSE needs rocPRIM. Need to add test!!!!

	for i in rocBLAS rocSPARSE rocSOLVER hipBLAS hipSPARSE
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i

		./install.sh -icd | tee $LOG_DIR/$CURR_BUILD.log
		popd
	done

	for i in rocRAND
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i

		./install -icd | tee $LOG_DIR/$CURR_BUILD.log
		popd
	done

	apt install libsqlite3-dev libbz2-dev half libboost-all-dev -y
	for i in MIOpen
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		rm -rf ./*
		cmake .. -DMIOPEN_BACKEND=OpenCL | tee $LOG_DIR/$CURR_BUILD
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		make install | tee -a $LOG_DIR/$CURR_BUILD
		popd
	done
        CURR_BUILD=HIP-Examples
        i=$CURR_BUILD
        echo $building $i
        pushd $ROCM_SRC_FOLDER/$i
        ./test_all.sh | tee $LOG_DIR/$CURR_BUILD.log
        popd

	# broken build on 4.1.

	for i in rocALUTION
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build ; cd build
		cmake .. -DSUPPORT_HIP=ON | tee $LOG_DIR/$CURR_BUILD.log
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
		popd
	done

        CURR_BUILD=HIP-Examples
        i=$CURR_BUILD
        echo $building $i
        pushd $ROCM_SRC_FOLDER/$i
        ./test_all.sh | tee $LOG_DIR/$CURR_BUILD.log
        popd

	apt install -y texinfo bison flex
        CURR_BUILD=ROCgdb
        echo $building $CURR_BUILD
        pushd $ROCM_SRC_FOLDER/$CURR_BUILD
        ./configure
	make -j$NPROC
        make
	popd

	pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
	rbuild build -d depend --cxx=/opt/rocm/llvm/bin/clang++

	CURR_BUILD=AMDMIGraphX
	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build; cd build
	CXX=/opt/rocm/llvm/bin/clang++ cmake .. | tee $LOG_DIR/$CURR_BUILD
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
	make install | tee -a $LOG_DIR/$CURR_BUILD
	popd

	CURR_BUILD=MIVisionX
	echo $building $i
	pushd $ROCM_SRC_FOLDER/$i
	mkdir build; cd build
	rm -rf ./*
	python MIVisionX-setup.py
	cmake .. | tee $LOG_DIR/$CURR_BUILD
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
	make install | tee -a $LOG_DIR/$CURR_BUILD
	popd

	#	sudo apt-get install scons mesa-common-dev libboost-all-dev rocprofiler-dev -y
	sudo apt-get install scons mesa-common-dev libboost-all-dev -y
	CURR_BUILD=RCP
	i=$CURR_BUILD
	echo $building $i
	pushd $ROCM_SRC_FOLDER/$i
	cd Build/Linux/ ; chmod 755 *sh
	#./build_rcp.sh skip-hsaprofiler | tee  $LOG_DIR/$CURR_BUILD
	./build_rcp.sh skip-hsaprofiler | tee  $LOG_DIR/$CURR_BUILD
	popd

else
	echo "Skipping over tested code..."
fi

if [[ $NON_REPO_ONLY == 1 ]] ; then
	echo "Installing non-rocm repo components build."

	pip3 install --user tensorflow-rocm --upgrade

else
	echo "Bypassing non-rocm repo components build."

fi

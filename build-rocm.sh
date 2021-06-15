REPO_ONLY=0
NON_REPO_ONLY=0
p1=$1
CONFIG_TEST=0
FAST_INSTALL=0
for var in "$@"
do
    if [[ $var == "fast" ]]  ; then
        echo fast installation specified: $var
        FAST_INSTALL=1
    fi
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
    echo "fast - install only core components for fast build finish."

    exit 0 ;
fi

if [[ -z $VERSION ]] ; then
	echo "You need to specify rocm version."
	exit 1
fi
LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
ROCM_SRC_FOLDER=~/ROCm-$VERSION
ROCM_INST_FOLDER=/opt/rocm-$VERSION.0/
LOG_SUMMARY=$LOG_DIR/build-summary.log
export PATH=$PATH:/opt/rocm-$VERSION.0/llvm/bin/

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

start=$SECONDS

mkdir -p $LOG_DIR
setup_root_rocm_softlink
setup_opt_rocm_softlink
echo "---" > $LOG_SUMMARY
echo "ROCm components that failed build." >> $LOG_SUMMARY
echo "If you don't see the name here, then build was successful." >> $LOG_SUMMARY

if [[ $REPO_ONLY == 0 ]] && [[ $NON_REPO_ONLY == 0 ]] ; then
	REPO_ONLY=1
	NON_REPO_ONLY=1
fi

echo "REPO_ONLY: $REPO_ONLY "
echo "NON_REPO_ONLY: $NON_REPO_ONLY"
echo "CONFIG_TEST: $CONFIG_TEST"

sleep 2

if [[ $CONFIG_TEST == 0 ]] && [[ $REPO_ONLY == 1 ]] ; then
	CURR_BUILD=llvm-project
	pushd $CURR_BUILD
	mkdir build ; cd build
	cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$VERSION.0/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC  | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
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
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make install | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

	mkdir -p "$COMGR/build"
	cd "$COMGR/build"
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build;$DEVICE_LIBS/build" .. | tee  -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC  | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make test | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make install | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	pushd $ROCM_SRC_FOLDER/ROCclr
	PWD=$ROCM_SRC_FOLDER/ROCclr/build
	OPENCL_DIR=$ROCM_SRC_FOLDER/ROCm-OpenCL-Runtime/
	ROCclr_DIR=$ROCM_SRC_FOLDER/ROCclr/
	OLDPWD=$ROCM_SRC_FOLDER/ROCclr

	mkdir build; cd build
	cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr .. | tee $LOG_DIR/ROCclr.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC install | tee -a $LOG_DIR/ROCclr.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	CURR_BUILD=HIP
	cd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build ; cd build
	cmake -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER/ROCclr/build;/opt/rocm/" .. | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make install | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

    if [[ $FAST_INSTALL -eq 0 ]] ; then	
    	CURR_BUILD=ROCm-OpenCL-Runtime
    	cd $ROCM_SRC_FOLDER/$CURR_BUILD
    	mkdir -p build; cd build
    	cmake -DUSE_COMGR_LIBRARY=ON -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER//ROCclr/build;/opt/rocm/" ..  | tee $LOG_DIR/$CURR_BUILD.log
    	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
    	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	make install | tee -a $LOG_DIR/$CURR_BUILD.log
    	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

    	CURR_BUILD=rccl
    	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    	./install.sh -idt | tee $LOG_DIR/$CURR_BUILD
    	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	popd
    fi

	# for rocr_debug_agent!!

	apt install gcc g++ make cmake libelf-dev libdw-dev -y

    if [[ $FAST_INSTALL -eq 0 ]] ; then	
	for i in rocm_smi_lib rocm_bandwidth_test rocminfo rocprofiler
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		cmake .. | tee $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make install | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done
    else
	for i in rocm_smi_lib rocm_bandwidth_test rocminfo rocprofiler rocr_debug_agent MIOpenGEMM half clang-ocl rocm-cmake  ROCR-Runtime/src ROCT-Thunk-Interface
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		cmake .. | tee $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make install | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done
    fi

    if [[ $FAST_INSTALL -eq 0 ]] ; then	
    	for i in  rocThrust
    	do
    		CURR_BUILD=$i
    		echo $building $i
    		pushd $ROCM_SRC_FOLDER/$i
    		mkdir build; cd build
    		rm -rf ./*
    		CXX=/opt/rocm/hip/bin/hipcc cmake .. | tee $LOG_DIR/$CURR_BUILD
    		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
    		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    		make install | tee -a $LOG_DIR/$CURR_BUILD
    		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    		popd
    	done
    fi

    if [[ $FAST_INSTALL -eq 0 ]] ; then	
	CURR_BUILD=ROCmValidationSuite
	pushd $ROCM_SRC_FOLDER/ROCmValidationSuite
	apt install libpciaccess-dev libpci-dev -y | tee  $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	cmake ./ -B./build | tee -a $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -C ./build | tee -a $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	for i in rocPRIM hipCUB
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		CXX=/opt/rocm/hip/bin/hipcc cmake -DBUILD_BENCHMARK=on .. | tee $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make install | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build ; cd build
	CXX=/opt/rocm/hip/bin/hipcc cmake -DBUILD_BENCHMARK=ON ../. | tee $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC install | tee -a $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	# rocSPARSE needs rocPRIM. Need to add test!!!!

	for i in rocBLAS rocSPARSE rocSOLVER hipBLAS hipSPARSE
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i

		./install.sh -icd | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

	for i in rocRAND
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i

		./install -icd | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

	apt install libsqlite3-dev libbz2-dev half libboost-all-dev -y
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	for i in MIOpen
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		rm -rf ./*
		cmake .. -DMIOPEN_BACKEND=OpenCL | tee $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make install | tee -a $LOG_DIR/$CURR_BUILD
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

	# broken build on 4.1.

	for i in rocALUTION
	do
		CURR_BUILD=$i
		echo $building $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build ; cd build
		cmake .. -DSUPPORT_HIP=ON | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

        CURR_BUILD=HIP-Examples
        i=$CURR_BUILD
        echo $building $i
        pushd $ROCM_SRC_FOLDER/$i
        ./test_all.sh | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
        popd

    fi # if fast_install = 0.

	apt install -y texinfo bison flex

    if [[ $FAST_INSTALL -eq 0 ]] ; then	
        CURR_BUILD=ROCgdb
        echo $building $CURR_BUILD
        pushd $ROCM_SRC_FOLDER/$CURR_BUILD
        ./configure
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	CURR_BUILD=AMDMIGraphX
	pip install https://github.com/RadeonOpenCompute/rbuild/archive/master.tar.gz
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	rbuild build -d depend --cxx=/opt/rocm/llvm/bin/clang++
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build; cd build
	CXX=/opt/rocm/llvm/bin/clang++ cmake .. | tee $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make install | tee -a $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	CURR_BUILD=MIVisionX
	echo $building $i
	pushd $ROCM_SRC_FOLDER/$i
	mkdir build; cd build
	python MIVisionX-setup.py
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	cmake .. | tee $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC | tee -a $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make install | tee -a $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	#	sudo apt-get install scons mesa-common-dev libboost-all-dev rocprofiler-dev -y
	CURR_BUILD=RCP
	sudo apt-get install scons mesa-common-dev libboost-all-dev -y
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	i=$CURR_BUILD
	echo $building $i
	pushd $ROCM_SRC_FOLDER/$i
	cd Build/Linux/ ; chmod 755 *sh
	#./build_rcp.sh skip-hsaprofiler | tee  $LOG_DIR/$CURR_BUILD
	./build_rcp.sh skip-hsaprofiler | tee  $LOG_DIR/$CURR_BUILD
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

    fi # if fast_install.
else
	echo "Skipping over tested code..."
fi

if [[ $NON_REPO_ONLY == 1 ]] && [[ $CONFIG_TEST == 0 ]]; then
	echo "Installing non-rocm repo components build."

	# install tensorflow.

	apt install python3-setuptools -y
	pip3 install --user tensorflow-rocm --upgrade

	# install pytorch profiler kineto:

	CURR_BUILD=pytorch-kineto
	mkdir -p ~/non-rocm-repo/
	pushd ~/non-rocm-repo/
	git clone --recursive https://github.com/pytorch/kineto.git
	cd kineto/libkineto/
	mkdir build ; cd build  
	cmake .. ; make -j`nproc` install | tee -a $LOG_DIR/$CURR_BUILD

else
	echo "Bypassing non-rocm repo components build."

fi
end=$SECONDS
duration=$(($end-$start))
echo "Build finished, build duration: $duration seconds."

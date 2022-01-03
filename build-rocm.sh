#Several variables dictate the speed and completeness of all builds:
# FAST_INSTALL - intended for building subset of components as quickly as possible.
# ESSENTIAL_INSTALL - intended for building only components whose binary package is not available (at least not found)
# So that user can install prebuild packages.When this option is one, suboption ESSENTIAL_INSTALL_PACKAGES - will
# install pre-built packages for all components.
#  - B: build
#  - I: install
#  - P: install pre-build package
#  - x: will not install
#  - TBD - to be determined, more work needed to determine.
#  - dontcare or DC - will install regardless of flag.
#  - dontcareN or DCN - will not install regardless of flag.

#---------------------------
# varname 		        | FAST_INSTALL	| ESSENTIAL_INSTALL 	|
# llvm			        | dontcare	| dontcare		|
# device-libs		    | dontcare	| dontcare		|
# comgr			        | DC		| DC			|
# rocclr		        | DC		| DC			|
# HIP			        | DC		| DC 			|
# ROCm-OpenCL-Runtime	| BI		| TBD 			|
# RCCL			        | BI 		| TBD			|
# rocm_smi_lib  	    | BI		| TBD 			|
# rocm_bandwidth_test 	| BI        | TBD           |
# rocminfo 		        | BI        | TBD           |
# rocprofiler		    | BI        | TBD           |
# rocr_debug_agent 	    | x	    	| TBD			|
# MIOpenGEMM 		    | x         | TBD           |
# half 			        | x         | TBD           |
# clang-ocl 		    | x         | TBD           |
# rocm-cmake 		    | x         | TBD           |
# ROCR-Runtime/src 	    | x         | TBD           |
# ROCT-Thunk-Interface	| x         | TBD           |
# roctracer		        | BI		| TBD			|
# rocThrust		        | x		    | TBD			|
# ROCmValidationSuite	| x		    | TBD			|
# rocPRIM		        | x		    | TBD			|
# hipCUB		        | x		    | TBD			|
# hipcc			        | x 	 	| TBD			|
# rocSPARSE 		    | x         | TBD           |
# rocSOLVER 		    | x         | TBD           |
# hipBLAS 		        | x         | TBD           |
# hipBLAS 		        | x         | TBD           |
# rocBLAS		        | x         | TBD           |
# rocRAND		        | x         | TBD           |  
# MIOpen		        | x         | TBD           |
# rocALUTION		    | x         | TBD           |
# rocGDB		        | x         | TBD           |
# AMDMIGRAPHx		    | x         | TBD           |
# MIVisionX		        | x         | TBD           |
# RCP(obsolete)         | x         | TBD           |
# 

REPO_ONLY=0
NON_REPO_ONLY=0
p1=$1
CONFIG_TEST=0
FAST_INSTALL=0
ESSENTIAL_INSTALL=0
CONFIG_BUILD_PACKAGE=""
CONFIG_BYPASS_LLVM=0
CONFIG_DISABLE_rocSOLVER=1
CONFIG_DISABLE_hipBLAS=1
t1=""
f2=""

OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
echo "OS_NAME: $OS_NAME"
case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

sleep 3
PKG_EXEC=apt

if [[ ! -z $CONFIG_BUILD_PACKAGE ]] ; then
	CONFIG_BUILD_PKGS_LOC=/rocm-packages/
	BUILD_TARGET=package
    INSTALL_SH_PACKAGE="-p"
	INSTALL_TARGET=package
	mkdir -p $CONFIG_BUILD_PKGS_LOC
else
	BUILD_TARGET=""
    INSTALL_SH_PACKAGE=""
	INSTALL_TARGET=install
fi
for var in "$@"
do
    if [[ $var == "llvmno" ]]  ; then
        echo bypass llvm: $var
       	CONFIG_BYPASS_LLVM=1
    fi
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

    if [[ ! -z `echo "$var" | grep "verminor="` ]]  ; then
        MINOR_VERSION=`echo $var | cut -d '=' -f2`
	echo "minor version: $MINOR_VERSION" ; sleep 5
    fi

    if [[ ! -z `echo "$var" | grep "pkg="` ]]  ; then
        PKG_EXEC=`echo $var | cut -d '=' -f2`
        echo Set pkg exec to $PKG_EXEC
    fi
done

echo Set pkg exec to $PKG_EXEC

if [[ $PKG_EXEC == "yum" ]] ; then echo "Installing epel-release ..." ; sleep 1 ;yum install epel-release gcc -y ; fi
$PKG_EXEC install python3-setuptools rpm -y

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
ROCM_INST_FOLDER=/opt/rocm-$VERSION.$MINOR_VERSION
LOG_SUMMARY=$LOG_DIR/build-summary.log
export PATH=$PATH:/opt/rocm-$VERSION.$MINOR_VERSION/llvm/bin/

function build_entry () {
    t2=$SECONDS
    if  [[ ! -z $t1 ]] ; then
        echo Build took $((t2-t1)) seconds 2>&1 | tee -a $LOG_SUMMARY
        echo "............................." 2>&1 | tee -a $LOG_SUMMARY
    else
        echo "Did not log previous build" 2>&1 | tee -a $LOG_SUMMARY
    fi
    L_CURR_BUILD=$1
    echo "............................." 2>&1 | tee -a $LOG_SUMMARY
    echo " Building entry: $L_CURR_BUILD" 2>&1 | tee -a $LOG_SUMMARY
    echo "............................." 2>&1 | tee -a $LOG_SUMMARY
    t1=$SECONDS
}

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
	if [[ $CONFIG_BYPASS_LLVM == 0 ]] ; then
		CURR_BUILD=llvm-project
        build_entry $CURR_BUILD
		pushd $CURR_BUILD
		mkdir build ; cd build
		cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/opt/rocm-$VERSION.0/llvm -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_PROJECTS="clang;lld;lldb;clang-tools-extra;compiler-rt" ../llvm 2>&1 | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC 2>&1 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
		popd
	else
		echo "Bypassing llvm..." ; sleep 5
	fi

	setup_root_rocm_softlink

	CURR_BUILD=ROCm-Device-Libs
	DEVICE_LIBS=$ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir -p "$DEVICE_LIBS/build"
	cd "$DEVICE_LIBS/build"
	cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$LLVM_PROJECT/build" .. | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	cp *deb *rpm $CONFIG_BUILD_PKGS_LOC/

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
	popd

	pushd $ROCM_SRC_FOLDER/ROCclr
	PWD=$ROCM_SRC_FOLDER/ROCclr/build
	OPENCL_DIR=$ROCM_SRC_FOLDER/ROCm-OpenCL-Runtime/
	ROCclr_DIR=$ROCM_SRC_FOLDER/ROCclr/
	OLDPWD=$ROCM_SRC_FOLDER/ROCclr

    CURR_BUILD=ROCclr
    build_entry $CURR_BUILD
	mkdir build; cd build
	cmake -DOPENCL_DIR="$OPENCL_DIR" -DCMAKE_INSTALL_PREFIX=/opt/rocm/rocclr .. | tee $LOG_DIR/ROCclr.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	#make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/ROCclr.log
	make -j$NPROC 2>&1 | tee -a $LOG_DIR/ROCclr.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	#make -j$NPROC $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/ROCclr.log
	make install 2>&1 | tee -a $LOG_DIR/ROCclr.log
	popd

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

	CURR_BUILD=HIP
    build_entry $CURR_BUILD
	cd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build ; cd build

    export HIPAMD_DIR="$(readlink -f hipamd.git)"
    export HIP_DIR="$(readlink -f hip.git)"
    export ROCclr_DIR="$(readlink -f ROCclr.git)"
    export OPENCL_DIR="$(readlink -f ROCm-OpenCL-Runtime.git)"
    echo HIPAMD_DIR: $HIPAMD_DIR, HIP_DIR: $HIP_DIR, ROCclr_DIR: $ROCclr_DIR, OPENCL_DIR: $OPENCL_DIR
    sleep 10

	cmake -DCMAKE_PREFIX_PATH="$ROCM_SRC_FOLDER/ROCclr/build;/opt/rocm/" .. | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC $BUILD_TARGET2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	make -j$NPROC 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	#make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	make install 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi

    if [[ $FAST_INSTALL -eq 0 ]] ; then	

    	CURR_BUILD=rccl
        build_entry $CURR_BUILD
   	    pushd $ROCM_SRC_FOLDER/$CURR_BUILD
    	./install.sh -idt $INSTALL_SH_PACKAGE | tee $LOG_DIR/$CURR_BUILD.log
    	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	popd
    fi

	# for rocr_debug_agent!!

	$PKG_EXEC install gcc g++ make cmake libelf-dev libdw-dev -y

    if [[ $FAST_INSTALL -eq 1 ]] ; then	
	for i in rocm_smi_lib rocm_bandwidth_test rocminfo rocprofiler
	do
		CURR_BUILD=$i
		build_entry $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		cmake .. | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done
    else
	for i in rocm_smi_lib rocm_bandwidth_test rocminfo rocprofiler rocr_debug_agent MIOpenGEMM half clang-ocl rocm-cmake  ROCR-Runtime/src ROCT-Thunk-Interface
	do
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
	done
    fi

    for i in roctracer
    do
        CURR_BUILD=$i
        build_entry $i
        $PKG_EXEC install rpm -y
        pip3 install cppheaderparser
        pushd $ROCM_SRC_FOLDER/$i
        ./build.sh
        popd
    done

    if [[ $FAST_INSTALL -eq 0 ]] ; then	
    	for i in  rocThrust
    	do
    		CURR_BUILD=$i
    		build_entry $i
    		pushd $ROCM_SRC_FOLDER/$i
    		mkdir build; cd build
    		rm -rf ./*
    		CXX=/opt/rocm/hip/bin/hipcc cmake .. | tee $LOG_DIR/$CURR_BUILD.log
    		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    		make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    		make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    		popd
    	done
    fi

    if [[ $FAST_INSTALL -eq 0 ]] ; then	
	CURR_BUILD=ROCmValidationSuite
	pushd $ROCM_SRC_FOLDER/ROCmValidationSuite
	$PKG_EXEC install libpciaccess-dev libpci-dev -y | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	cmake ./ -B./build 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -C ./build 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	popd

	for i in rocPRIM hipCUB
	do
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
	done

	pushd $ROCM_SRC_FOLDER/$CURR_BUILD
	mkdir build ; cd build
	CXX=/opt/rocm/hip/bin/hipcc cmake -DBUILD_BENCHMARK=ON ../. | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j$NPROC $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	popd

	# rocSPARSE needs rocPRIM. Need to add test!!!!

	for i in rocSPARSE rocSOLVER hipBLAS hipSPARSE
	do
        echo "GG: CONFIG_DISABLE_$i: $((CONFIG_DISABLE_$i))"
        sleep 10
		if [[ $((CONFIG_DISABLE_$i)) == 1 ]] ; then
 			echo "Bypassing $i build..." >> $LOG_SUMMARY
		else
			CURR_BUILD=$i
			build_entry $i
			pushd $ROCM_SRC_FOLDER/$i
    		./install.sh -icd | tee $LOG_DIR/$CURR_BUILD.log
    		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    		popd
		fi
	done
	for i in rocBLAS
	do
		CURR_BUILD=$i
		build_entry $i
		pushd $ROCM_SRC_FOLDER/$i

		./install.sh -icd -logic asm_full | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

	for i in rocRAND
	do
		CURR_BUILD=$i
		build_entry $i
		pushd $ROCM_SRC_FOLDER/$i

		./install -icd | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

	$PKG_EXEC install libsqlite3-dev libbz2-dev half libboost-all-dev -y
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	for i in MIOpen
	do
		CURR_BUILD=$i
		build_entry $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build; cd build
		rm -rf ./*
        cmake -DMIOPEN_BACKEND=OpenCL -DMIOPEN_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ .. | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

	# broken build on 4.1.

	for i in rocALUTION
	do
		CURR_BUILD=$i
		build_entry $i
		pushd $ROCM_SRC_FOLDER/$i
		mkdir build ; cd build
		cmake .. -DSUPPORT_HIP=ON | tee $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		make -j$NPROC $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
		if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
		popd
	done

        CURR_BUILD=HIP-Examples
        i=$CURR_BUILD
        build_entry $i
        pushd $ROCM_SRC_FOLDER/$i
        ./test_all.sh | tee $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
        popd

    fi # if fast_install = 0.

	$PKG_EXEC install -y texinfo bison flex

    echo "FAST_INSTALL: $FAST_INSTALL"
    if [[ $FAST_INSTALL == 0 ]] ; then	
        CURR_BUILD=ROCgdb
        build_entry $CURR_BUILD
        pushd $ROCM_SRC_FOLDER/$CURR_BUILD
        ./configure
    	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	make -j$NPROC
    	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	popd

    	CURR_BUILD=AMDMIGraphX
        build_entry $CURR_BUILD
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

        # Commented out because it seems obsolete, github shows last commit 2019. Keep for a while and delete 
        # this code.
    	#	sudo apt-get install scons mesa-common-dev libboost-all-dev rocprofiler-dev -y
    	#CURR_BUILD=RCP
    	#sudo apt-get install scons mesa-common-dev libboost-all-dev -y
    	#if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	#i=$CURR_BUILD
    	#build_entry $i
    	#pushd $ROCM_SRC_FOLDER/$i
    	#cd Build/Linux/ ; chmod 755 *sh
    	#./build_rcp.sh skip-hsaprofiler | tee  $LOG_DIR/$CURR_BUILD.log
    	#if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
    	#popd

    fi # if fast_install.
else
	echo "Skipping over tested code..."
fi

if [[ $NON_REPO_ONLY == 1 ]] && [[ $CONFIG_TEST == 0 ]]; then
	echo "Installing non-rocm repo components build."

	mkdir -p ~/non-rocm-repo/
	pushd ~/non-rocm-repo/
    
	# install tensorflow.

	$PKG_EXEC install python3-setuptools -y
	pip3 install --user tensorflow-rocm --upgrade

	# install pytorch profiler kineto:

	CURR_BUILD=pytorch-kineto
    build_entry $CURR_BUILD
	git clone --recursive https://github.com/pytorch/kineto.git
	cd kineto/libkineto/
	mkdir build ; cd build  
	cmake .. 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	make -j`nproc` $BUILD_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
	if [[ $? -ne 0 ]] ; then echo "$CURR_BUILD fail" >> $LOG_SUMMARY ; fi
	make -j`nproc` $INSTALL_TARGET 2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    cd ../../..
    pwd

    CURR_BUILD=rccl-tests
    build_entry $CURR_BUILD
    git clone https://github.com/ROCmSoftwarePlatform/rccl-tests.git
    cd $CURR_BUILD 
    ./install.sh |2>&1 | tee -a $LOG_DIR/$CURR_BUILD.log
    cd ..

    CURR_BUILD=grpc
    build_entry $CURR_BUILD
    git clone -b v1.28.1 https://github.com/grpc/grpc
    echo "curr dir 1: " ; pwd
    cd $CURR_BUILD 
    git submodule update --init
    mkdir build ; cd build
    cmake -DgRPC_INSTALL=ON -DBUILD_SHARED_LIBS=ON  ..
    make -j`nproc` 
    make $INSTALL_TARGET    
    cd ../..

    # onnx + prereq (protobuf)
    
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

    CURR_BUILD=onnx
    build_entry $CURR_BUILD
    export CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON"
    git clone --recursive https://github.com/onnx/onnx.git
    cd $CURR_BUILD 
    set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON
    pip3 install -e.
    cd ..

    popd
else
	echo "Bypassing non-rocm repo components build."

fi
end=$SECONDS
duration=$(($end-$start))
echo "Build finished, build duration: $duration seconds or $((duration/60)) minutes."

files=`tree -fi | grep \.deb$ | grep -v _CPack_Packages > deb.log`
for i in files; do
    echo $i 
    cp $i $CONFIG_BUILD_PKGS_LOC/
done
	

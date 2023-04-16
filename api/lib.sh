function set_os_type() {
   OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
   echo "OS_NAME: $OS_NAME"
   case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      ;;
   "CentOS Stream")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      return 0
      ;;
   *)
      echo "Unsupported O/S, exiting..."
      PKG_EXEC=""
      return 1
     ;;
    esac
}
function install_python() {
    #Python defs.
    set_os_type

    if [[ $? -ne 0 ]] ; then return 1 ; fi

    CURR_VER=`python3 --version  | cut -d ' ' -f2`
    PYTHON_VER_MAJOR=3.9
    PYTHON_VER_MINOR=10

    PYTHON_VER=$PYTHON_VER_MAJOR.$PYTHON_VER_MINOR
    PYTHON_FULL_NAME=Python-$PYTHON_VER
    PYTHON_TAR=$PYTHON_FULL_NAME.tgz

    if [[ $CURR_VER == $PYTHON_VER ]] ; then
        echo "Current installed version is same as the one being installed..., exiting"
        return 0
    else
        echo "Installing..."
    fi

    if [[ $OS_NAME == *"Centos"* ]] ; then
        sudo $PKG_EXEC -y install epel-release
        sudo $PKG_EXEC update -y
        sudo $PKG_EXEC groupinstall "Development Tools" -y
        sudo $PKG_EXEC install openssl-devel libffi-devel bzip2-devel -y
    elif [[ $OS_NAME == *"Ubuntu"* ]] ; then
        sudo $PKG_EXEC-get update -y
        sudo $PKG_EXEC install openssl-dev libffi-dev bzip2-dev
    fi
    
    wget -nc https://www.python.org/ftp/python/$PYTHON_VER/$PYTHON_TAR
    tar -xvf $PYTHON_TAR
    cd $PYTHON_FULL_NAME

    if [[ $? -ne 0 ]] ; then
        echo "Can not cd into $PYTHON_VER directory..."
        exit 1
    fi
    ./configure --enable-optimizations
    sudo make -j`nproc` install

    echo "Testing the installation..."
    python$PYTHON_VER_MAJOR --version
    if [[ $? -ne 0 ]] ; then
        echo "Unable to find 3.9"
    fi
    PATH_PYTHON_U=`which python$PYTHON_VER_MAJOR`
    echo "new path: $PATH_PYTHON_U"
    rm -rf /usr/bin/python
    echo ln -s $PATH_PYTHON_U /usr/bin/python
    ln -s $PATH_PYTHON_U /usr/bin/python
    rm -rf /usr/bin/python3
    ln -s /usr/bin/python /usr/bin/python3
    cd ..
    return 0
}

function install_conda() {
    
    # Anaconda defs.

    FILE_ANACONDA=Anaconda3-2021.11-Linux-x86_64.sh
    PREFIX_ANACONDA=/Anaconda3
    CONFIG_UPGRADE_ANACONDA=1
    EXEC_PATH_CONDA=$PREFIX_ANACONDA/bin/conda

    wget -nc https://repo.anaconda.com/archive/$FILE_ANACONDA
    ret=$? ; echo $ret.
    if [[ $ret -ne 0 ]] ; then
        echo "code: $ret. Download failure for $FILE_ANACONDA, check the url."
        exit 1
    fi
    chmod  777 $FILE_ANACONDA
    mkdir -p $PREFIX_ANACONDA
    if [[ $CONFIG_UPGRADE_ANACONDA=="1" ]] ; then
        echo "Installing/upgrading regardless of existing installation..."
        ./$FILE_ANACONDA  -u -b -p $PREFIX_ANACONDA
    else
        echo "Installing only if it is not installed in $PREFIX_ANACONDA location..."
        ./$FILE_ANACONDA  -b -p $PREFIX_ANACONDA
    fi    


    #if [[ `cat ~/.bashrc | grep PATH | grep $PREFIX_ANACONDA` ]] ; then
    #    echo "Inserting path onto bashrc in case reboot next time. "
    #    echo "export PATH=$PATH:$PREFIX_ANACONDA/bin" >> ~/.bashrc
    #else
    #    echo "conda path is already defined in bashrc."
    #fi
    ln -s $EXEC_PATH_CONDA /usr/bin/conda
    echo "Testing the installation..."
    $EXEC_PATH_CONDA

    if [[ $? -ne 0 ]] ; then
        echo "Can not find $EXEC_PATH_CONDA."
        exit 1
    fi  
}

CONFIG_TEST=0
FAST_INSTALL=0
ESSENTIAL_INSTALL=0
CONFIG_BUILD_PACKAGE=0
CONFIG_BYPASS_LLVM=0
CONFIG_DISABLE_rocSOLVER=1
CONFIG_DISABLE_hipBLAS=1

function install_pip_libs_centos() {
    for i in cppheaderparser pyyaml ; do
        echo =======================
        pip3 install $i
    done
    
}
function install_python2() {
    #Python defs.

    CURR_VER=`python3 --version  | cut -d ' ' -f2`
    PYTHON_VER_MAJOR=3.9
    PYTHON_VER_MINOR=10

    PYTHON_VER=$PYTHON_VER_MAJOR.$PYTHON_VER_MINOR
    PYTHON_FULL_NAME=Python-$PYTHON_VER
    PYTHON_TAR=$PYTHON_FULL_NAME.tgz

    if [[ $CURR_VER == $PYTHON_VER ]] ; then
        echo "Current installed version is same as the one being installed..., exiting"
        return 0
    else
        echo "Installing..."
    fi

    sudo yum -y install epel-release
    sudo yum update -y
    sudo yum groupinstall "Development Tools" -y

    sudo yum install openssl-devel libffi-devel bzip2-devel -y
    wget -nc https://www.python.org/ftp/python/$PYTHON_VER/$PYTHON_TAR
    tar -xvf $PYTHON_TAR
    cd $PYTHON_FULL_NAME

    if [[ $? -ne 0 ]] ; then
        echo "Can not cd into $PYTHON_VER directory..."
        exit 1
    fi
    ./configure --enable-optimizations
    sudo make -j`nproc` install

    echo "Testing the installation..."
    python$PYTHON_VER_MAJOR --version
    if [[ $? -ne 0 ]] ; then
        echo "Unable to find 3.9"
    fi
    PATH_PYTHON_U=`which python$PYTHON_VER_MAJOR`
    echo "new path: $PATH_PYTHON_U"
    rm -rf /usr/bin/python
    echo ln -s $PATH_PYTHON_U /usr/bin/python
    ln -s $PATH_PYTHON_U /usr/bin/python

    rm -rf /usr/bin/python3
    ln -s /usr/bin/python /usr/bin/python3
    cd ..
}

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

LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
#ROCM_SRC_FOLDER=~/ROCm-$VERSION
ROCM_INST_FOLDER=/opt/rocm-$VERSION.$MINOR_VERSION
LOG_SUMMARY=$LOG_DIR/build-summary.log
LOG_SUMMARY_L2=$LOG_DIR/build-summary-l2.log

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

VERSION="5.2"
MINOR_VERSION="0"
VERSION=$vermajor
MINOR_VERSION=$verminor
mkdir /log/rocmbuild/ -p
ROCM_SRC_FOLDER=/root/gg/git/ROCm-$VERSION/
echo "ROCM_SRC_FOLDER: $ROCM_SRC_FOLDER, minor version: $MINOR_VERSION"
export ROCM_SRC_FOLDER=$ROCM_SRC_FOLDER

if [[ $CONFIG_BUILD_PACKAGE -ne 0 ]] ; then
    echo "will build packages..."
    CONFIG_BUILD_PKGS_LOC=/rocm-packages/
    BUILD_TARGET=package
    INSTALL_SH_PACKAGE="-p"
    INSTALL_TARGET=package
    mkdir -p $CONFIG_BUILD_PKGS_LOC
else
    echo "will not build packages..."
    BUILD_TARGET=""
    INSTALL_SH_PACKAGE=""
    INSTALL_TARGET=install
fi

OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
echo "OS_NAME: $OS_NAME"
case "$OS_NAME" in
   "Ubuntu")
        echo "Ubuntu is detected..."
        PKG_EXEC=apt
        apt-get update
        for i in python3-pip sqlite3 libsqlite3-dev libbz2-dev nlohmann-json-dev half libboost-all-dev python-msgpack pybind11-dev numactl libudev1 libudev-dev chrpath pciutils pciutils-dev libdw libdw-dev 
        do  
            $PKG_EXEC install $i  -y 2>&1 | tee -a $LOG_SUMMARY_L2 
            if [[ $? -ne 0 ]] ; then 
                echo "Failed to install $i" | tee -a $LOG_SUMMARY_L2 ; 
            fi 
        done
      #gem install json
        
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      $PKG_EXEC install --skip-broken sqlite-devel sqlite half boost boost-devel gcc make cmake  numactl numactl-devel dpkg pciutils-devel mesa-libGL-devel libpciaccess-dev libpci-dev -y  2>&1 | tee -a $LOG_SUMMARY_L2
      $PKG_EXEC install gcc g++ make cmake libelf-dev libdw-dev numactl numactl-devel -y
      install_pip_libs_centos
      ;;
   "CentOS Stream")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      $PKG_EXEC install gcc g++ make cmake libelf-dev libdw-dev numactl numactl-devel -y
      $PKG_EXEC install --skip-broken sqlite-devel sqlite half boost boost-devel gcc make cmake  numactl numactl-devel dpkg pciutils-devel mesa-libGL-devel libpciaccess-dev libpci-dev -y  2>&1 | tee -a $LOG_SUMMARY_L2
      install_pip_libs_centos
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac 

install_python

# p1 - env varname
# p2 - env value.
# example: export_bashrc ROCM_DIR /opt/rocm
# example: export_bashrc ROCM_DIR $ROCM_DIR (if ROCM_DIR is defined)

function export_bashrc() {
    env_name=$1
    env_value=$2

    export $env_name=$env_value

    if [[ -z $env_name ]] || [[ -z env_value ]] ; then echo "env_name or env_value is empty" ; return 1; fi

    [[ `grep "export.*$env_name" ~/.bashrc` ]] && \
    sed -i --expression "s/export.*${env_name}.*/export ${env_name}=${env_value}/g" ~/.bashrc || \
    echo "export $env_name=$env_value" | tee -a ~/.bashrc

}

function export_bashrc_delim_alt() {
    env_name=$1
    env_value=$2

    export $env_name=$env_value

    if [[ -z $env_name ]] || [[ -z env_value ]] ; then echo "env_name or env_value is empty" ; return 1; fi

    [[ `grep "export.*$env_name" ~/.bashrc` ]] && \
    sed -i --expression "s@export.*${env_name}.*@export ${env_name}=${env_value}@g" ~/.bashrc || \
    echo "export $env_name=$env_value" | tee -a ~/.bashrc
}

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
    set -x
    #Python defs.
    set_os_type

    if [[ $? -ne 0 ]] ; then return 1 ; fi

    if [[ `which python3` ]] ; then 
        CURR_VER=`python3 --version  | cut -d ' ' -f2`
    else
        echo "Unable to find python3 path, assuming not installed..."
    fi

    if [[ -z $1 ]] ; then
        echo "install_python: don't know what version to install."
        return 1
    fi
    PYTHON_VER=$1

    echo "Installing python version: $PYTHON_VER..."

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
    if [[ $? -ne 0 ]] ; then echo "Unable to wget $PYTHON_TAR" ; return 1 ; fi
    tar -xvf $PYTHON_TAR
    cd $PYTHON_FULL_NAME

    if [[ $? -ne 0 ]] ; then
        echo "Can not cd into $PYTHON_VER directory..."
        exit 1
    fi
    ./configure --enable-optimizations
    if [[ $? -ne 0 ]] ; then echo "configure fail..." ; exit 1 ; fi

    sudo make -j`nproc` install
    if [[ $? -ne 0 ]] ; then echo "configure fail..." ; exit 1 ;fi
    sudo make  install
    if [[ $? -ne 0 ]] ; then echo "configure fail..." ; exit 1 ;fi

    echo "Testing the installation..."
    python$PYTHON_VER_MAJOR --version
    if [[ $? -ne 0 ]] ; then
        echo "Unable to find 3.9"
    fi
    PATH_PYTHON_U=`which python$PYTHON_VER_MAJOR`
    echo "new path: $PATH_PYTHON_U"
    mv /usr/bin/python /usr/bin/python.old
    echo ln -s $PATH_PYTHON_U /usr/bin/python
    ln -s $PATH_PYTHON_U /usr/bin/python
    mv /usr/bin/python3 /usr/bin/python3.old
    ln -s /usr/bin/python /usr/bin/python3
    cd ..
    return 0
}

function install_conda() {
    
    # Anaconda defs.

   
    if [[ $1 ]] ; then
        FILE_ANACONDA=$1
    else
        FILE_ANACONDA=Anaconda3-2021.11-Linux-x86_64.sh
    fi
    echo "Installing the version: $FILE_ANACONDA" 
    sleep 5

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

install_cmake() {
    p1=$1
    if [[ -z $p1 ]] ; then
        VERSION=3.16.8
    else
        VERSION=$p1
    fi

    OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
    echo "OS_NAME: $OS_NAME"
    case "$OS_NAME" in
       "Ubuntu")
          echo "Ubuntu is detected..."
          PKG_EXEC=apt
          sudo apt install build-essential libssl-dev -y
          ;;
       "CentOS Stream")
          echo "CentOS is detected..."
          PKG_EXEC=yum
          ln -s /usr/bin/python3  /usr/bin/python
          yum groupinstall "Development Tools" -y
          yum install openssl-devel -y
          ;;
       *)
         echo "Unsupported O/S, exiting..." ; exit 1
         ;;
    esac

    wget https://github.com/Kitware/CMake/releases/download/v$VERSION/cmake-$VERSION.tar.gz
    tar -zxvf cmake-$VERSION.tar.gz
    cd cmake-$VERSION
    ./bootstrap
    make  -j8
    sudo make install 
    ret=`cat ~/.bashrc | grep CMAKE_ROOT`

    if [[ -z $ret ]] ; then echo "export CMAKE_ROOT=`which cmake`" >> ~/.bashrc ; fi

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

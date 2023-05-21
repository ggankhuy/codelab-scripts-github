echo "common.sh entered..."

function print_single_bar() {
    local i
    for i in {1..50} ; do echo -ne "-" ; done
    echo ""
}

function print_double_bar() {
    local i
    for i in {1..50} ; do echo -ne "=" ; done
    echo ""
}

function set_os_type() {
   OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
   echo "OS_NAME: $OS_NAME"
   case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      PKG_EXT=deb
      ln -s /usr/bin/python3  /usr/bin/python
      ;;
   "CentOS Stream")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      PKG_EXT=rpm
      return 0
      ;;
   *)
      echo "Unsupported O/S, exiting..."
      PKG_EXEC=""
      PKG_EXT=""
      return 1
     ;;
    esac
}

function install_packages() {
    if [[ -z $PKG_EXEC  ]] ; then echo "PKG_EXEC is not defined. Call set_os_type first!" ; return 1 ; fi
    for i in $@; do
        print_single_bar
        echo "installing $i..."
        $PKG_EXEC install -y $i
    done
}

function install_pip_libs() {
    for i in $@; do
        print_single_bar
        echo "installing $i..."
        pip3 install $i
    done
    
}
function install_python() {
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
    t1=$SECONDS
    L_CURR_BUILD=$1
    print_double_bar | tee -a $LOG_SUMMARY
    echo "Building entry: $L_CURR_BUILD" 2>&1 | tee -a $LOG_SUMMARY
    print_double_bar
}

function build_exit() {
    t2=$SECONDS
    echo "Build took $((t2-t1)) seconds" 2>&1 | tee -a $LOG_SUMMARY
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

function rocm_source_dw() {
    p1=$1
    SUDO=sudo
    CONFIG_VERSION=4.1
    DATE=`date +%Y%m%d-%H-%M-%S`
    HOME_DIR=`pwd`
    if [[ -z $p1 ]] ; then
        echo "Version not specified."
        return 200
    else
        CONFIG_VERSION=$p1
    fi

    set_os_type

    git config --global user.email "you@example.com"
    git config --global user.name "Your Name"
    git config --global color.ui false
    DIR_NAME=$HOME_DIR/ROCm-$CONFIG_VERSION

    if [[ -d $DIR_NAME ]] ; then
        echo "Directory $DIR_NAME exists, moving to $DUR_NAME-$DATE "
        $SUDO mv $DIR_NAME $DIR_NAME-$DATE
    fi
    mkdir $DIR_NAME

    if [[ $? -ne 0 ]] ; then
    	echo "Directory is already there. Verify it is deleted or renamed before continue." ; return 1
    fi

    pushd  $DIR_NAME
    mkdir -p ~/bin/
    echo "install repo..."
    $SUDO $PKG_EXEC install curl -y && $SUDO curl https://storage.googleapis.com/git-repo-downloads/repo | $SUDO tee ~/bin/repo
    $SUDO chmod a+x ~/bin/repo
    echo "repo init..."
    $SUDO ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-$CONFIG_VERSION.x
    echo "repo sync..."
    $SUDO ~/bin/repo sync
    echo "ROCm source is downloaded to $DIR_NAME"
    echo "push $DIR_NAME to get there..."
    popd
}

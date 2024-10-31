##set -x
echo "common.sh entered..."

# Following settings are only local to this file: common.sh.
# Do not put any settings that is applicable to APIs/functions
# script outside of this shell script.

OPTION_EXIT_ON_ROCM_SOURCE_CHECKOUT=0

ERROR_VERSION=300
ERROR_ROCM_SRC_REPO_INIT=301
ERROR_ROCM_SRC_REPO_SYNC=302

function print_single_bar() {
    local i
    set +x
    for i in {1..50} ; do echo -ne "-" ; done
    echo ""
    set -x
}

function print_double_bar() {
    local i
    set +x
    for i in {1..50} ; do echo -ne "=" ; done
    echo ""
    set -x
}

#   This function is also called by python code and parses its stdout, therefore
#   Do not alter any output preceding with PY: string!

function set_os_type() {
   OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
   echo "OS_NAME: $OS_NAME"
   case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      echo "PY:PKG_EXEC=apt"
      echo "PY:PKG_EXT=deb"
      PKG_EXEC=apt
      PKG_EXT=deb
      ln -s /usr/bin/python3  /usr/bin/python
      ;;
   "Red Hat Enterprise Linux")
      echo "CentOS is detected..."
      echo "PY:PKG_EXEC=yum"
      echo "PY:PKG_EXT=rpm"
      PKG_EXEC=yum
      PKG_EXT=rpm
      ln -s /usr/bin/python3  /usr/bin/python
      return 0
      ;;
   "CentOS Stream")
      echo "CentOS is detected..."
      echo "PY:PKG_EXEC=yum"
      echo "PY:PKG_EXT=rpm"
      PKG_EXEC=yum
      PKG_EXT=rpm
      return 0
      ;;
    "openSUSE Leap")
      echo "OpenSUSE Leap is detected..."
      echo "PY:PKG_EXEC=rpm"
      echo "PY:PKG_EXT=rpm"
      PKG_EXEC=zypper
      PKG_EXT=rpm
      ln -s /usr/bin/python3  /usr/bin/python
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
        echo "installing $i..."
        case "$PKG_EXEC" in
        "yum")
            $PKG_EXEC install -y $i
            ;;
        "apt")
            $PKG_EXEC install -y $i
            ;;
        "zypper")
            $PKG_EXEC -n install $i
            ;;
        *)
            echo "Unsupported/unknown PKG installer: $PKG_EXEC, exiting..."
      ;;    
    esac

        print_single_bar
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
    echo -ne "$L_CURR_BUILD," | tee -a $LOG_SUMMARY_CSV
    sleep 3
    print_double_bar
}

function build_exit() {
    t2=$SECONDS
    L_BUILD_ENTRY=$1
    L_BUILD_RESULT=$2
    if [[ -z $L_BUILD_RESULT  ]] ; then L_BUILD_RESULT="UNKNOWN" ; fi
    echo "Build took $((t2-t1)) seconds" 2>&1 | tee -a $LOG_SUMMARY
    echo -ne "$((t2-t1))," | tee -a $LOG_SUMMARY_CSV
    echo "$L_BUILD_RESULT" | tee -a $LOG_SUMMARY_CSV
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
        return $ERROR_VERSION
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
   case "$PKG_EXEC" in
   "yum")
        $SUDO $PKG_EXEC install curl -y --allowerasing && $SUDO curl https://storage.googleapis.com/git-repo-downloads/repo | $SUDO tee ~/bin/repo
      ;;
   "apt")
        $SUDO $PKG_EXEC install curl -y && $SUDO curl https://storage.googleapis.com/git-repo-downloads/repo | $SUDO tee ~/bin/repo
      ;;
   "zypper")
        $SUDO $PKG_EXEC -n install curl && $SUDO curl https://storage.googleapis.com/git-repo-downloads/repo | $SUDO tee ~/bin/repo
      ;;
   *)
        echo "Unsupported or unknown package installer: $PKG_EXEC"
        return $ERROR_ROCM_SRC_REPO_INIT
      ;;
    esac

    $SUDO chmod a+x ~/bin/repo
    echo "repo init..."
    $SUDO ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-$CONFIG_VERSION.x
    if [[ $? -ne 0 ]] ; then
        echo "Error: Unable to do initialize repo."
        return $ERROR_ROCM_SRC_REPO_INIT
    fi
    echo "repo sync..."
    $SUDO ~/bin/repo sync

    if [[ $? -ne 0 ]] && [[ $OPTION_EXIT_ON_ROCM_SOURCE_CHECKOUT -ne 0 ]] ; then
        echo "Error: Unable to perform repo sync."
        return $ERROR_ROCM_SRC_REPO_SYNC
    else
        echo "Warning: Error occurred while performing repo sync. Continuing anyways..., Ctrl+c to terminate to inspect if necessary"
        sleep 5
    fi
    echo "ROCm source is downloaded to $DIR_NAME"
    echo "push $DIR_NAME to get there..."
    popd
}

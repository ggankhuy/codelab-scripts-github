CONFIG_TEST=0
FAST_INSTALL=0
ESSENTIAL_INSTALL=0
CONFIG_BUILD_PACKAGE=0
CONFIG_BYPASS_LLVM=0
CONFIG_DISABLE_rocSOLVER=1
CONFIG_DISABLE_hipBLAS=1

source sh/common.sh

OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
echo "OS_NAME: $OS_NAME"
case "$OS_NAME" in
   "Ubuntu")
        echo "Ubuntu is detected..."
        PKG_EXEC=apt
        SHELL=bash
        apt-get update -y
        for i in git-lfs cmake python3-pip sqlite3 libsqlite3-dev libbz2-dev nlohmann-json-dev half libboost-all-dev python-msgpack pybind11-dev numactl libudev1 libudev-dev chrpath pciutils pciutils-dev libdw libdw-dev 
        do  
            echo "Installing $i...."
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
      SHELL=sh
      $PKG_EXEC install --skip-broken sqlite-devel sqlite half boost boost-devel gcc make cmake  numactl numactl-devel dpkg pciutils-devel mesa-libGL-devel libpciaccess-dev libpci-dev -y  2>&1 | tee -a $LOG_SUMMARY_L2
      $PKG_EXEC install git-lfs gcc g++ make cmake libelf-dev libdw-dev numactl numactl-devel -y
      install_pip_libs_centos
      ;;
   "CentOS Stream")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      SHELL=sh
      $PKG_EXEC install git-lfs gcc g++ make cmake libelf-dev libdw-dev numactl numactl-devel -y
      $PKG_EXEC install --skip-broken sqlite-devel sqlite half boost boost-devel gcc make cmake  numactl numactl-devel dpkg pciutils-devel mesa-libGL-devel libpciaccess-dev libpci-dev -y  2>&1 | tee -a $LOG_SUMMARY_L2
      install_pip_libs_centos
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac 



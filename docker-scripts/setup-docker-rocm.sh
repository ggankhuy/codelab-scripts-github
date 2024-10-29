    # Use this script to install ROCm onto Centos 9 stream docker container.
# Currently only RHEL9.2 version is supported.
# All other configuration will likely to cause fail or unpredictable result.
# Parameter conflicts are not checked. Usage menu has most 3 commont usage: ga, internal mainline, internal release along with 
# combination of parameters needed. Any other combination of parameters are currently untested and could fail or cause unpredictable
# result.


printHelp() {
    set +x
    echo "Usage: parameters:"
    echo "Note that currently there is not compatility check between different parameters. It is up to user to provide"
    echo "right combination of parameters. Using wrong combination or incompatible parameters are untested and will have "
    echo "untested or unpredictable output and could SUT in uncertain state."
    echo "--ga: use GA general availability repo.radeon.com."
    echo "--int: use internal version, artifactory."
    echo "--rocm-build: rocm build number."
    echo "--rocm-ver: rocm version."
    echo "--mainline: use mainline branch for internal build."
    echo "--release: use release branch for internal build." 
    echo "--amdgpu-build: amdgpu build number."
    echo "--no-dkms: bypass dkms installation."
    echo "--no-install: skip whole installation, only do prereq steps"
<<<<<<< HEAD
    echo "--nogpgcheck: bypass gpg check" 
=======
>>>>>>> 0d653266ab71cfdb5b0a9c1b5d3e0e6224399362
    echo "Usage: examples"
    echo "$0 --ga --rocm-ver=6.0 / use ga version of rocm version 6.0"
    echo "$0 --int --mainline --rocm-build=13435 --rocm-ver=6.1 --amdgpu-build=1720120 / use internal version of rocm"
    echo "$0 --int --release --rocm-build=91 --rocm-ver=6.0 / use internal version of rocm"
    exit 0
}

if [[ -z $1 ]] ; then
    printHelp
fi
set -x
for var in "$@"
do
    echo var: $var

    case "$var" in 
        "--help")
            printHelp
            ;;
        *--no-install*)
            p_no_install=1
            ;;
        *--ga*)
            p_ga=1
            ;;
        *--int*)
            p_int=1
            ;;
        *--rocm-ver=*)
            rocmversion=`echo $var | awk -F'=' '{print $2}'`
            ;;
        *--mainline*)
            p_mainline=1
            ;;
        *--release*)
            p_release=1
            ;;
        *--rocm-build=*)
            p_rocm_build=`echo $var | awk -F'=' '{print $2}'`

            ;;
        *--amdgpu-build=*)
            p_amdgpu_build=`echo $var | awk -F'=' '{print $2}'`
            ;;
        *--no-dkms*)
            p_dkms="--no-dkms"
            ;;
<<<<<<< HEAD
        *--nogpgcheck)
            p_nogpgcheck="--nogpgcheck"
            ;;
=======
>>>>>>> 0d653266ab71cfdb5b0a9c1b5d3e0e6224399362
        *--root)
            p_root=1
            ;;
        *)
            echo "Unknown parameter: $var. Exiting..."
            exit 1
            ;;
    esac
done

set -x 
if [[ $p_root ]] && [[ -z $USER ]] ; then 
    USER=$SUDO_USER
    HOME=/home/$USER
else
    USER=/root
    HOME=/root
fi

yum update -y ; yum install cmake git tree nano wget g++ python3-pip sudo -y
dnf install epel-release epel-next-release -y ; dnf config-manager --set-enabled crb ; dnf install epel-release epel-next-release -y
cd /$HOME/extdir ; mkdir gg; cd gg ; mkdir git log wget back transit ; cd git ; echo "cd `pwd`" >> /$HOME/.bashrc

if [[ $p_no_install == 1 ]] ; then  exit 0 ; fi

rhel_ver=9.2

for i in rocm amdgpu ; do 
    yum repository-packages $i remove -y ; #if installed through online repository.
    yum remove $i -y # if installed through offline bundle script.
    rm -rf /etc/yum.repos.d/$i*.repo
done

# this apparently not working, try using above.
amdgpu-install --uninstall -y

for i in amdgpu-install amdgpu-install-internal ; do 
    yum remove $i -y
done

mkdir -p /$HOME/extdir/gg/wget
pushd /$HOME/extdir/gg/wget 

rm -rf amdgpu-install*

echo "rocmversion: $rocmversion"
if [[ ! -z $p_ga ]] ; then
    url_amdgpu="http://repo.radeon.com/amdgpu-install/$rocmversion/rhel/$rhel_ver/"
    wget --mirror -L -np -nH -c -nv --cut-dirs=6 -A "*.rpm" -P ./ $url_amdgpu
fi

if [[ ! -z $p_int ]] ; then
    amdgpu_file_name="amdgpu-install-internal-$rocmversion"
    amdgpu_file_name=$amdgpu_file_name"_9-1.noarch.rpm"
    url_amdgpu="http://artifactory-cdn.amd.com/artifactory/list/amdgpu-rpm/rhel/$amdgpu_file_name"
    wget $url_amdgpu
fi

echo "url_amdgpu: $url_amdgpu"

#if [[ -z $url_amdgpu ]] ; then echo "url_amdgpu is not set." ; exit 1 ; fi
#if [[ ! -z $? ]] ; then "echo wget unsuccessful..."  ; exit 1 ; fi

amdgpu_installer_rpm_path=`find . -name "amdgpu-install*" | head -1`
<<<<<<< HEAD
yum install $amdgpu_installer_rpm_path  -y $p_nogpgcheck
=======
yum install $amdgpu_installer_rpm_path  -y
>>>>>>> 0d653266ab71cfdb5b0a9c1b5d3e0e6224399362
popd
echo "$rhel_ver" > /etc/dnf/vars/amdgpudistro

if [[ ! -z $p_int ]] ; then
    if [[ ! -z $p_mainline ]] ;  then
        url="http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-dkms-no-npi-hipclang-$p_rocm_build/"
        amdgpu-repo --amdgpu-build=$p_amdgpu_build --rocm-build=compute-rocm-dkms-no-npi-hipclang/$p_rocm_build
    fi
    if [[ ! -z $p_release ]] ; then 
        amdgpu-repo --amdgpu-build=$p_amdgpu_build --rocm-build=compute-rocm-rel-$rocmversion/$p_rocm_build
    fi
fi 

<<<<<<< HEAD
amdgpu-install --usecase=rocm $p_dkms -y $p_nogpgcheck
=======
amdgpu-install --usecase=rocm $p_dkms -y
>>>>>>> 0d653266ab71cfdb5b0a9c1b5d3e0e6224399362






if [[ -z $1 ]] ; then
    echo "Usage: "
    echo "$0 --ga --version=6.0 / use ga version of rocm version 6.0"
    echo "$0 --int --mainline --build=13435 --amdgpu=1720120 / use internal version of rocm"
    echo "$0 --int --branch --build=91 --version --amdgpu=1720120 / use internal version of rocm"

    exit 0
fi

yum update -y ; yum install cmake git tree nano wget g++ python3-pip -y
dnf install epel-release epel-next-release -y ; dnf config-manager --set-enabled crb ; dnf install epel-release epel-next-release -y
cd ~/extdir ; mkdir gg; cd gg ; mkdir git log wget back transit ; cd git ; echo "cd `pwd`" >> ~/.bashrc

set -x 

for var in "$@"
do
    echo var: $var

    if [[ $var == *"--ga"* ]]  ; then
        p_ga=1
    fi
    if [[ $var == *"--int"* ]]  ; then
        p_int=1
    fi
    if [[ $var == *"--version="* ]]  ; then
        p_version=`echo $var | cut -d '=' -f2`
    fi
    if [[ $var == *"--mainline"* ]]  ; then
        p_mainline=1
    fi
    if [[ $var == *"--branch"* ]]  ; then
        p_branch=1
    fi
    if [[ $var == *"--build="* ]]  ; then
        p_build=`echo $var | cut -d '=' -f2`
    fi
    if [[ $var == *"--amdgpu="* ]]  ; then
        p_amdgpu=`echo $var | cut -d '=' -f2`
    fi
done

rocm_ver=$p_version
rocm_build_no=$p_build
rhel_ver=9.2

if [[ ! -z $p_ga ]] ; then
    url="http://repo.radeon.com/amdgpu-install/$rocm_ver/rhel/$rhel_ver/"
fi

if [[ ! -z $p_int ]] ; then
    url="http://mkmartifactory.amd.com/artifactory/amdgpu-rpm-local/rhel/$rhel_ver/builds/1720129/x86_64/a/amdgpu-install*"
    if [[ ! -z $p_mainline ]] ;  then
        amdgpu-repo --amdgpu-build=$p_amdgpu --rocm-build=compute-rocm-dkms-no-npi-hipclang/$rocm_build_no
        #url="http://compute-artifactory.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-dkms-no-npi-hipclang-$rocm_build_no/"
    fi
    if [[ ! -z $p_branch ]] ; then 
        amdgpu-repo --amdgpu-build=1739896 --rocm-build=compute-rocm-rel-$rocm_ver/$rocm_build_no
        #url="http://compute-cdn.amd.com/artifactory/list/rocm-osdb-rhel-9.x/compute-rocm-rel-$rocm_ver-$rocm_build_no/
    fi
fi

pushd ~/extdir/gg/wget 
wget --mirror -L -np -nH -c -nv --cut-dirs=6 -A "*.rpm" -P ./ $url
yum install ./amdgpu-install*.rpm -y
popd
echo "$rhel_ver" > /etc/dnf/vars/amdgpudistro

amdgpu-install --usecase=rocm --no-dkms -y












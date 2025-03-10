set -x 

OPTION_BUILD_RVS_SUPPORT=1
OPTION_CLEAN_BUILD_RDC=1
OPTION_CREATE_PKG=1
GRPC_PROTOC_ROOT=/opt/grpc

source ../../api/lib.sh 
set_os_type

#centos : gcrp part working.
#ubuntu: in test.

case "$OS_NAME" in
"Ubuntu")
  PKG_LIST="doxygen libpcap-dev libcap-dev automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl libyaml-cpp-dev libabsl-dev"
  ;;
"CentOS Stream")
  echo "CentOS is detected..."
  PKG_LIST="doxygen libpcap-devel libcap-devel automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-devel libgtest-devel clang-5.0 libc++-devel curl libyaml-cpp-devel libabsl-devel"
  ;;
*)
  echo "Unsupported O/S, exiting..."
  PKG_EXEC=""
  return 1
  ;;
esac

for i in $PKG_LIST ; do
    echo "Installing $i ..." ; sudo yum install $i -y
done

CMAKE_PARAMS="-DROCM_DIR=/opt/rocm -DGRPC_ROOT=$GRPC_PROTOC_ROOT -DCMAKE_INSTALL_PREFIX=/opt/rocm -DBUILD_RVS=on" 

if [[ $OPTION_CREATE_PKG == 1 ]] ;          then CMAKE_PARAMS="$CMAKE_PARAMS -DCPACK_GENERATOR=DEB" ; fi
if [[ $OPTION_BUILD_RVS_SUPPORT == 1 ]] ;   then apt install rocm-validation-suite -y ; CMAKE_PARAMS="$CMAKE_PARAMS -DBUILD_RVS=ON" ; fi

if [[ ! -z rdc ]] ; then
    git clone https://github.com/ROCm/rdc
fi
pushd rdc
pwd
mkdir -p build
cd build

if [[ $OPTION_CLEAN_BUILD_RDC -eq 1 ]] ; then rm -rf build ; fi


CMAKE_PREFIX_PATH=/root/extdir/gg/git/codelab-scripts/rocm-scripts/rdc/grpc/build/third_party/protobuf/cmake/protobuf/ \
cmake $CMAKE_PARAMS ..

#cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT" ..
#cmake  -DGRPC_ROOT="$GRPC_ROOT" -DBUILD_STANDALONE=off -DBUILD_RVS=Off .. 

#if [[ -z $? ]] ; then make -j $(nproc) install else ; exit 1; fi
make -j $(nproc)
popd

RDC_LIB_DIR=/opt/rocm/rdc/lib
GRPC_LIB_DIR=/opt/grpc/lib
echo -e "${GRPC_LIB_DIR}\n${GRPC_LIB_DIR}64" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
echo -e "${RDC_LIB_DIR}\n${RDC_LIB_DIR}64" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
ldconfig

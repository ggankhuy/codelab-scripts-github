set -x 
OPTION_CLEAN_BUILD_GRPC=0
source ../../api/lib.sh

if [[ $OPTION_CLEAN_BUILD_GRPC -eq 1 ]] ; then rm -rf build ; fi

set_os_type

#centos : gcrp part working.
#ubuntu: in test.

case "$OS_NAME" in
"Ubuntu")
  PKG_LIST="doxygen libpcap-dev"
  ;;
"CentOS Stream")
  echo "CentOS is detected..."
  PKG_LIST="doxygen libpcap-devel"
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

if [[ ! -d grpc ]] ; then 
    git clone -b v1.61.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
fi

pushd  grpc
export GRPC_ROOT=/opt/grpc
cmake -B build \
    -DgRPC_INSTALL=ON \
    -DgRPC_BUILD_TESTS=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_BUILD_TYPE=Release && make -C build -j $(nproc) && sudo make -C build install
echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf
popd



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
   "CentOS Linux")
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

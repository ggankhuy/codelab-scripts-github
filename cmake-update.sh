p1=$1
if [[ -z $p1 ]] ; then
	VERSION=3.16.8
else
	VERSION=$p1
fi
sudo apt install build-essential libssl-dev -y
wget https://github.com/Kitware/CMake/releases/download/v$VERSION/cmake-$VERSION.tar.gz
tar -zxvf cmake-$VERSION.tar.gz
cd cmake-$VERSION
./bootstrap
make  -j8
sudo make install 
ret=`cat ~/.bashrc | grep CMAKE_ROOT`

if [[ -z $ret ]] ; then echo "export CMAKE_ROOT=`which cmake`" >> ~/.bashrc ; fi

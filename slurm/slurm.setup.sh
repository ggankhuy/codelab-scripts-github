echo "OS_NAME: $OS_NAME"
case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      sudo yum install bzip2 zlib pkgconf openssl-devel -y
      ln -s /usr/bin/python3  /usr/bin/python
      ;;
   "CentOS Stream")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      sudo yum install bzip2 zlib pkgconf openssl-devel -y
      ln -s /usr/bin/python3  /usr/bin/python
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

# following seems worked for munge build:
origin  https://github.com/dun/munge.git (fetch)
origin  https://github.com/dun/munge.git (push)

  122  libtoolize --force
  123  aclocal
  124  autoheader
  125  autoconf
  126  ./configure
  127  $ autoreconf -vif
  128  yum install autoreconf
  129  yum whatprovides autoreconf
  130  yum install -y autoconf
  131  autoreconf -i
  132  ./configure 
  133  ./configure                   --prefix=/usr                 --sysconfdir=/etc             --localstatedir=/var          --runstatedir=/run  
  134  ./configure                   --prefix=/usr                 --sysconfdir=/etc             --localstatedir=/var          --runstatedir=/run
  135  ./configure                   --prefix=/usr                 --sysconfdir=/etc             --localstatedir=/var        
  136  make -j8
  137  make check
  138  make install
  139  history

git clone https://github.com/SchedMD/slurm.git
cd slurm
./configure 
make `nproc`
make install
cd ..



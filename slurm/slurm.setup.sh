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
git clone https://github.com/dun/munge.git

cd munge
autoreconf -i
./configure              \
     --prefix=/usr            \
     --sysconfdir=/etc        \
     --localstatedir=/var     \
     --runstatedir=/run       \
make -j`nproc` 
make install


git clone https://github.com/SchedMD/slurm.git
cd slurm
chmod 755 configure
adduser slurm
echo "slurm:amd1234" | chpasswd slurm
./configure --prefix=/slurm --sysconfdir=/slurm-conf
make `nproc`
make install
cp ./slurmctld.service /etc/systemd/system/slurmctld.service
systemctl enable slurmctld
systemctl start slurmctld
cp ./slurmdbd.service /etc/systemd/system/slurmdbd.service
systemctl enable slurmdbd
systemctl start slurmdbd
cd ..



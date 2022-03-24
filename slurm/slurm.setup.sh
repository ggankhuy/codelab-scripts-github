OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
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
     --with-runstatedir=/run
#For some reason this is not working.

#     --runstatedir=/run       \
make -j`nproc` 
make install
cd ..

adduser munge
echo "munge:amd1234" | chpasswd munge
mkdir -p /run/munge
chown -R munge: /etc/munge/ /var/log/munge/ /var/lib/munge/ /run/munge/
sudo -u munge /usr/sbin/mungekey --verbose
# should create /etc/munge/munge.key
chmod 600 /etc/munge/munge.key
systemctl enable munge
systemctl start munge

git clone https://github.com/SchedMD/slurm.git
cd slurm
chmod 755 configure
adduser slurm
echo "slurm:amd1234" | chpasswd slurm
./configure --prefix=/slurm --sysconfdir=/slurm-conf
make `nproc`
make install
cd ..
pwd
cp slurmctld.service /etc/systemd/system/slurmctld.service
systemctl enable slurmctld
systemctl start slurmctld
cp slurmdbd.service /etc/systemd/system/slurmdbd.service
systemctl enable slurmdbd
systemctl start slurmdbd

if [[ -z `cat ~/.bashrc | grep slurm/bin` ]] ; then
	echo "inserting path /slurm/bin to bashrc..."
	echo "export PATH=$PATH:/slurm/bin" >> ~/.bashrc
else
	echo "path /slurm/bin already in bashrc..."
fi



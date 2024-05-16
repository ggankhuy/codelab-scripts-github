SUDO=sudo
$SUDO mkdir -p /miniconda3
$SUDO wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
$SUDO chmod 755 ./miniconda.sh
$SUDO bash ./miniconda.sh -b -u -p /miniconda3
$SUDO rm -rf ./miniconda.sh
if [[ -z `cat ~/.bashrc | grep export | grep miniconda3` ]] ; then
    echo "export PATH=$PATH:/miniconda3/bin" >> ~/.bashrc
fi

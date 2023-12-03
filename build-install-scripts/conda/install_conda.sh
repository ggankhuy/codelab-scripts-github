mkdir -p /miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
chmod 755 ./miniconda.sh
bash ./miniconda.sh -b -u -p /miniconda3
rm -rf ./miniconda.sh
echo "export PATH=$PATH:/miniconda3/bin" >> ~/.bashrc

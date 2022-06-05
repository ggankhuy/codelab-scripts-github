yum groupinstall "Development Tools" -y
python3 -m pip3 install --upgrade pip
yum update
yum install zlib python-devel -y
pip3 install neuralnet matplotlib pandas sklearn Pillow scipy==1.2.0 cloudpickle pyparsing  requests idna --upgrade
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2

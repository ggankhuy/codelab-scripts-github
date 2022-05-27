yum groupinstall "Development Tools" -y
python3 -m pip3 install --upgrade pip
yum update
yum install zlib -y
pip3 install neuralnet matplotlib pandas sklearn Pillow
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2

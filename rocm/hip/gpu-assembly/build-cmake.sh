sudo mkdir build
cd build
sudo rm -rf ./*
sudo cmake -DCMAKE_PREFIX_PATH=/opt/rocm/hip/lib ..
make


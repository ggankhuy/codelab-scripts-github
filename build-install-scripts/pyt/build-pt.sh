# Anaconda defs.
source ../lib.sh

CONFIG_UPGRADE_ANACONDA=1

# building pytorch
echo "Building pytorch..."
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git submodule sync

git submodule update --init --recursive --jobs 0

for i in setuptools pip distlib pyyaml  ; do
    pip3 install --upgrade $i
done

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python tools/amd_build/build_amd.py 2>&1  | tee build-pytorch.log
PYTORCH_ROCM_ARCH=gfx908 python setup.py install


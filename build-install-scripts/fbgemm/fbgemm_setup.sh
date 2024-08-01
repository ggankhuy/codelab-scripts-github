#!/bin/bash

## This script only supports FBGEMM setup with ROCm ver 6.2 ~ 6.0

set -x

# Check ROCm version
rocm_ver=$(cat /opt/rocm/.info/version)

if [[ "${rocm_ver:0:5}" == "6.2.0" || ${rocm_ver:0:5} == "6.1.0" ]]; then
    nightly_torch="https://download.pytorch.org/whl/nightly/rocm6.1"
else
    nightly_torch="https://download.pytorch.org/whl/nightly/rocm6.0"
fi

# Install nightly torch
pip3 install --pre torch --index-url $nightly_torch

# Clone FBGEMM and sync submodule
git clone https://github.com/pytorch/FBGEMM.git 
pushd FBGEMM
pwd
git submodule sync
git submodule update --init --recursive

# Install requirements.txt
pushd fbgemm_gpu
pip install -r requirements.txt

# Build FBGEMM
export MAX_JOBS='nproc' && gpu_arch="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')" && export PYTORCH_ROCM_ARCH=$gpu_arch && git clean -dfx && python setup.py --package_variant rocm -DHIP_ROOT_DIR=/opt/rocm -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA" build develop 2>&1 | tee build.log

popd
popd



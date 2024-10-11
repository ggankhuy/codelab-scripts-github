#!/bin/bash

## This script only supports FBGEMM setup with ROCm ver 6.2 ~ 6.0

set -x

usage() {
  echo "Usage: $0 --repo <pytorch/rocm> [--fbgemm-branch <FBGEMM-branch>] "
  exit 1
}


# Parse arguments
FBGEMM_BRANCH=""
REPO=""

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --fbgemm-branch) FBGEMM_BRANCH="$2"; shift ;;
        --repo) REPO="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$REPO" ]]; then
  usage
fi

# Determine the repository URL based on the --repo parameter
if [[ "$REPO" == "pytorch" ]]; then
    REPO_URL="https://github.com/pytorch/FBGEMM.git"
    echo "Using https://github.com/pytorch/FBGEMM.git"
elif [[ "$REPO" == "rocm" ]]; then
    REPO_URL="https://github.com/ROCm/FBGEMM.git"
    echo "Using https://github.com/ROCm/FBGEMM.git"
else
    echo "Warning: Unsupported repo '$REPO'. Please use 'pytorch' or 'rocm'."
    usage
    exit 1
fi

# Check ROCm version
rocm_ver=$(cat /opt/rocm/.info/version)

if [[ "${rocm_ver:0:5}" == "6.3.0" || "${rocm_ver:0:5}" == "6.2.0" || ${rocm_ver:0:5} == "6.1.0" ]]; then
    # latest available is 6.1
    nightly_torch="https://download.pytorch.org/whl/nightly/rocm6.1"
else
    nightly_torch="https://download.pytorch.org/whl/nightly/rocm6.0"
fi

# Install nightly torch
pip3 install --pre torch --index-url $nightly_torch

# Clone FBGEMM and sync submodule
git clone $REPO_URL
pushd FBGEMM

# Checkout the specified branch if provided
if [ -n "$FBGEMM_BRANCH" ]; then
    if git ls-remote --heads $REPO_URL $FBGEMM_BRANCH | grep -q $FBGEMM_BRANCH; then
        git checkout "$FBGEMM_BRANCH"
    else
        echo "Warning: Branch '$FBGEMM_BRANCH' does not exist in the '$REPO' repository. Exiting."
        exit 1
    fi
fi

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



## If wish to run UVM test
# cd test
# export FBGEMM_TEST_WITH_ROCM=1
# export HIP_LAUNCH_BLOCKING=1
# export HSA_XNACK=1
# python -m pytest -v -rsx -s -W ignore::pytest.PytestCollectionWarning uvm/uvm_test.py

## If wish to build experimental tests
# ./fbgemm_setup.sh --repo pytorch
# cp fbgemm.patch FBGEMM/
# cd FBGEMM
# patch -p1 < fbgemm.patch
# cd fbgemm_gpu
# export MAX_JOBS='nproc' && gpu_arch="$(/opt/rocm/bin/rocminfo | grep -o -m 1 'gfx.*')" && export PYTORCH_ROCM_ARCH=$gpu_arch && git clean -dfx && python setup.py --package_variant rocm -DHIP_ROOT_DIR=/opt/rocm -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA" build develop 2>&1 | tee build.log 

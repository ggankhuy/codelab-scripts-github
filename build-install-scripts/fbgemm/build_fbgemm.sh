set -x

CONFIG_INSTALL_PYTHON=0
CONFIG_INSTALL_TORCH=1
CONFIG_INSTALL_LLVM=0

env_name=`echo $CONDA_DEFAULT_ENV | awk "{print $2}"`
pwd=`pwd`
mkdir log
if [[ -z $env_name ]] ; then
    echo "Make sure conda environment is activated"
    exit 1
else
    echo "env_name: $env_name"
fi

if [[ $CONFIG_INSTALL_PYTHON -eq 1 ]] ; then
    #fbgemm dictates 12.0 but it is not available, so choosing closes one: 3.12.0
    python_version=3.12.0
    python_version_curr=`python3 --version | awk "{print $2}"`

    if [[ $python_version != $python_version_curr ]] ; then
        echo "python version does not match: $python_version [required], $python_version_curr [current]"
        echo "will try reinstall"
        pushd ../py/
        ./install_python.sh $python_version 2>&1 | tee $pwd/log/step1.install_python.log
        if [[ $? -ne 0 ]] ; then
            echo "install_python.sh failed..."
            popd    
            exit 1
        fi
        popd
    else
        echo "python_version: $python_version"
    fi

    # recheck.

    if [[ $python_version != $python_version_curr ]] ; then
        echo "python version does not match: $python_version [required], $python_version_curr [current]"
        echo "This is after attempt to install version $python_version so giving up."
        exit 1
    else
        echo "python_version: $python_version"
    fi
fi # install python

conda run pip install --upgrade pip
conda run python -m pip install pyOpenSSL>22.1.0

for i in hipify-clang miopen-hip miopen-hip-devel; do
    yum install $i -y
done

gcc_version=10.4.0
conda install -c conda-forge -y gxx_linux-64=${gcc_version} sysroot_linux-64=2.17
libcxx_path=`find /usr -name libstdc++.so.6`

if [[ -z $libcxx_path ]] ; then
    echo 'Unable to find libcxx_path' ; exit 1 
fi

# Print supported for GLIBC versions
objdump -TC "${libcxx_path}" | grep GLIBC_ | sed 's/.*GLIBC_\([.0-9]*\).*/GLIBC_\1/g' | sort -Vu | cat

# Print supported for GLIBCXX versions
objdump -TC "${libcxx_path}" | grep GLIBCXX_ | sed 's/.*GLIBCXX_\([.0-9]*\).*/GLIBCXX_\1/g' | sort -Vu | cat

if [[ CONFIG_INSTALL_LLVM -eq 1 ]] ; then
    llvm_version=15.0.7
    conda install -c conda-forge -y \
    clangxx=${llvm_version} \
    libcxx \
    llvm-openmp=${llvm_version} \
    compiler-rt=${llvm_version}
fi

# Append $CONDA_PREFIX/lib to $LD_LIBRARY_PATH in the Conda environment

ld_library_path=$(conda run -n ${env_name} printenv LD_LIBRARY_PATH)
conda_prefix=$(conda run -n ${env_name} printenv CONDA_PREFIX)
conda env config vars set -n ${env_name} LD_LIBRARY_PATH="${ld_library_path}:${conda_prefix}/lib"

# Set NVCC_PREPEND_FLAGS in the Conda environment for Clang to work correctly as the host compiler
## GG: this step is broken:
#usage: conda [-h] [-v] [--no-plugins] [-V] COMMAND ...
#conda: error: unrecognized arguments: -Xcompiler -std=c++20 -Xcompiler -stdlib=libstdc++ -ccbin -allow-unsupported-compiler"
#(fbgemm) [root@localhost fbgemm]# nano -w build_fbgemm.sh
#commenting out for now.

#conda env config vars set -n ${env_name} NVCC_PREPEND_FLAGS=\"-std=c++20 -Xcompiler -std=c++20 -Xcompiler -stdlib=libstdc++ -ccbin ${clangxx_path} -allow-unsupported-compiler\"

#compiler synliks

conda_prefix=$(conda run -n ${env_name} printenv CONDA_PREFIX)

ln -sf "${path_to_either_gcc_or_clang}" "$(conda_prefix)/bin/cc"
ln -sf "${path_to_either_gcc_or_clang}" "$(conda_prefix)/bin/c++"

# other build tools

conda install -n ${env_name} -y \
    click \
    cmake \
    hypothesis \
    jinja2 \
    make \
    ncurses \
    ninja \
    numpy \
    scikit-build \
    wheel

# install torch

if [[ $CONFIG_INSTALL_TORCH -eq 1 ]] ; then
    conda install -n ${env_name} -y pytorch -c pytorch-nightly
    # Ensure that the package loads properly
    conda run -n ${env_name} python -c "import torch.distributed"

    # Verify the version and variant of the installation
    conda run -n ${env_name} python -c "import torch; print(torch.__version__)"
fi

if [[ $CONFIG_INSTALL_TRITON -eq 1 ]] ; then
    # pytorch-triton repos:
    # https://download.pytorch.org/whl/nightly/pytorch-triton/
    # https://download.pytorch.org/whl/nightly/pytorch-triton-rocm/

    # The version SHA should follow the one pinned in PyTorch
    # https://github.com/pytorch/pytorch/blob/main/.ci/docker/ci_commit_pins/triton.txt
    conda run -n ${env_name} pip install --pre pytorch-triton==3.0.0+dedb7bdf33 --index-url https://download.pytorch.org/whl/nightly/
    conda run -n ${env_name} python -c "import triton"
fi

FBGEMM_VERSION=v0.8.0
#v0.8.0-release
git clone --recursive -b ${FBGEMM_VERSION}-release https://github.com/pytorch/FBGEMM.git fbgemm_${FBGEMM_VERSION}
pushd fbgemm_${FBGEMM_VERSION}/fbgemm_gpu
pip install -r requirements.txt
python setup.py clean

export package_name=fbgemm_gpu_rocm
# Set the Python version tag.  It should follow the convention `py<major><minor>`,
# e.g. Python 3.12 -> py312
export python_tag=py312

# Determine the processor architecture
export ARCH=$(uname -m)

# Set the Python platform name for the Linux case
export python_plat_name="manylinux2014_${ARCH}"
# For the macOS (x86_64) case
export python_plat_name="macosx_10_9_${ARCH}"
# For the macOS (arm64) case
export python_plat_name="macosx_11_0_${ARCH}"
# For the Windows case
export python_plat_name="win_${ARCH}"

# gen ai build:

CONFIG_FBGEMM_BUILD_VARIANT=genai
CONFIG_FBGEMM_BUILD_VARIANT=rocm
case $CONFIG_FBGEMM_BUILD_VARIANT in 
    genai)
       package_variant=genai
    ;;
    rocm)
       package_variant=rocm
    ;;
    *)
    echo "Unsupported build variant: $CONFIG_FBGEMM_BUILD_VARIANT"
    exit 1
    popd
esac

# Build and install the library into the Conda environment
# original genai variant command for cuda.
#python setup.py install \
#    --$package_variant=genai \
#    --nvml_lib_path=${NVML_LIB_PATH} \
#    --nccl_lib_path=${NCCL_LIB_PATH} \
#    -DTORCH_CUDA_ARCH_LIST="${cuda_arch_list}"

#modified genai variant (if defined) for rocm, if it works.

export ROCM_PATH=/opt/rocm/

if [[ -z $ROCM_PATH ]] ; then
    echo "rocm path is not found in $ROCM_PATH"
    exit 1
    popd
fi
python setup.py install \
    --package_variant=$package_variant \
    -DHIP_ROOT_DIR="${ROCM_PATH}" \
    -DCMAKE_C_FLAGS="-DTORCH_USE_HIP_DSA" \
    -DCMAKE_CXX_FLAGS="-DTORCH_USE_HIP_DSA"


popd



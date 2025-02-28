# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 

CONFIG_DEBUG_TORCH=0
CONFIG_DEBUG=0
CONFIG_MAGMA_REBUILD=1 # unused for now.
source ./lib_bash.sh
function CHECK_ERR() {
    ret=$?
    if [[ $ret -ne 0 ]] ; then echo "Error. Code: $ret" ; exit 1; fi
}

function list_mkl_info() {
    if [[ $CONFIG_DEBUG_TORCH -eq  1 ]] ; then
        echo "[DBG: checkpoint $1]------------ list_mkl_info ----------------"
        set -x
        ls -l /home/ggankhuy/miniconda3_src/pkgs | grep mkl
        ls -l /home/ggankhuy/miniconda3_src/$CONDA_ENV_NAME | grep mkl
        conda list | grep mkl
        pip3 list | grep mkl
        echo "DBG: ------------ list_mkl_info ----------------"
    fi
}

[[ $? -ne 0 ]] && exit 1

SOFT_LINK=1

# set up ulimit
LIMIT="/etc/security/limits.conf"
SEARCH_STRING="* soft nofile 1048576"
SEARCH_STRING_2="* hard nofile 1048576"
SEARCH_STRING_3="* soft memlock unlimited"
SEARCH_STRING_4="* hard memlock unlimited"

if ! grep -qF "$SEARCH_STRING" "$LIMIT" && ! grep -qF "$SEARCH_STRING_2" "$LIMIT" && ! grep -qF "$SEARCH_STRING_3" "$LIMIT" && ! grep -qF "$SEARCH_STRING_4" "$LIMIT"; then
  sudo sed -i '/# End of file/i \
  * soft nofile 1048576\n\
  * hard nofile 1048576\n\
  * soft memlock unlimited\n\
  * hard memlock unlimited' "$LIMIT"
fi

if [[ ! -f $LLAMA_PREREQ_PKGS.tar ]]; then 
    echo "$LLAMA_PREREQ_PKGS.tar does not exist." 
    exit 1
fi

tar -xvf  $LLAMA_PREREQ_PKGS.tar
pushd $LLAMA_PREREQ_PKGS

    # Force torch to be installed first. 

    torchwhl_path=`find . -name 'rocm_torch*.tar' | head -1`
    torchwhl=`basename $torchwhl_path`
    dirname="torch"
    echo $dirname
    mkdir $dirname ; 
        pushd $dirname
        pwd
        ln -s ../$torchwhl .
        tar -xvf ./$torchwhl
        pip3 install ./*.whl
        CHECK_ERR
        popd
    echo $torchwhl

    for i in *tar ; do 
        set +x
        echo "DBG: --------     Installing $i wheel package... ---------"
        set -x
        dirname=`echo $i | awk '{print $1}' FS=. `
        if [[ $dirname == "rocm_torch" ]] ; then
            continue
        fi
        mkdir $dirname ; 
            pushd $dirname
            pwd
            ln -s ../$i .
            tar -xvf ./$i 
            pip3 install ./*.whl
            CHECK_ERR
            popd
    done
popd

list_mkl_info 1
conda install mkl-service mkl -y
list_mkl_info 2
pip3 install mkl 
list_mkl_info 3
tar -xf $LLAMA_PREREQ_PKGS.tar
pwd
ls -l 

pushd $LLAMA_PREREQ_PKGS
mkdir log
bash install.sh 2>&1 | sudo tee log/install.log
popd
sudo ln -s `sudo find /opt -name clang++` /usr/bin/
if [[ -z `which clang++` ]] ; then echo "Error: can not setup or find clang++ in default path" ; exit 1 ; fi

[[ $CONFIG_MAGMA_REBUILD -eq 1 ]] && rm -rf magma
#git clone https://bitbucket.org/icl/magma.git 
git clone https://github.com/icl-utk-edu/magma.git
pushd magma 
find . -name libmagma.so
[[ $? -eq 0 ]] || exit 1

BASHRC=~/.bashrc
BASHRC_EXPORT=./export.md
ROCM_PATH=/opt/rocm/

ls -l $BASHRC

export_bashrc_delim_alt ROCM_PATH $ROCM_PATH

# build robust mkl so path using pip paths.

# This did not work, so commented out for now.
#MKLROOT_1=`pip3 show -f mkl | grep Location: | awk '{print $NF}'`
#MKLROOT=$MKLROOT_1
#MKLROOT_2=`pip3 show -f mkl | grep libmkl_intel_lp64 | awk '{print $NF}'` 
#MKLROOT_FULL=${MKLROOT_1}/${MKLROOT_2}
#MKLROOT=`dirname $MKLROOT_FULL`
#for i in {0..2}; do
#    MKLROOT=`dirname $MKLROOT`
#done

export_bashrc_delim_alt MKLROOT $MKLROOT
CONDA_PKG_CACHE_DIR=`conda info | grep  "package cache" | head -1  | awk '{print $NF}'`
CONDA_PKG_CACHE_DIR_MKL=`ls -l $CONDA_PKG_CACHE_DIR | grep "mkl-[0-9]" | grep -v "\.conda" | head -1 | awk '{print $NF}'`
CONDA_PKG_CACHE_PATH=$CONDA_PKG_CACHE_DIR/$CONDA_PKG_CACHE_DIR_MKL
MKLROOT=$CONDA_PKG_CACHE_PATH
if [[ ! -d $MKLROOT ]] ; then
    echo "Path does not exist for MKLROOT, can not continue: MKLROOT: $MKLROOT"
    exit 1
fi

# setup wheels in the package;

# magma section

PWD=`pwd`
export_bashrc_delim_alt MAGMA_HOME $PWD
export_bashrc MKLROOT $MKLROOT
export_bashrc_delim_alt ROCM_PATH $ROCM_PATH
cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo "LIBDIR += -L\$(MKLROOT)/lib" >> make.inc
echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,\$(ROCM_PATH)/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,\$(MAGMA_HOME)/lib" >> make.inc
echo "DEVCCFLAGS += --amdgpu-target=gfx942" >> make.inc
# build MAGMA
make -f make.gen.hipMAGMA -j 
HIPDIR=$ROCM_PATH GPU_TARGET=gfx942 make lib -j 2>&1 | tee ../log/env.$CONDA_DEFAULT_ENV.make.magma.log

popd

if [[ $SOFT_LINK == 1 ]] ; then
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        ln -s \
        $CONDA_PKG_CACHE_PATH/lib/$i.so.2 \
        $CONDA_PKG_CACHE_PATH/lib/$i.so.1
    done
else
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        rm -rf $CONDA_PKG_CACHE_PATH/lib/$i.so.1
        cp \
        $CONDA_PKG_CACHE_PATH/lib/$i.so.2 \
        $CONDA_PKG_CACHE_PATH/lib/$i.so.1
    done
fi

# following does not work for python.  even though ldcache includes those paths.
chmod 755 *sh
echo "Use following cmd to run:"
echo 'LD_LIBRARY_PATH=$MKLROOT/lib:$MAGMA_HOME/lib ./run_llama2_70b.sh'

echo "$MKLROOT/lib:$CONDA_PKG_CACHE_PATH/lib" | sudo tee /etc/ld.so.conf.d/mkl.conf
echo "$MAGMA_HOME/lib" | sudo tee /etc/ld.so.conf.d/magma.conf
ls -l /etc/ld.so.conf.d/

export_bashrc_delim_alt LD_LIBRARY_PATH $MKLROOT/lib:$MAGMA_HOME/lib

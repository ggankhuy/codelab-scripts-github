# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 

source ./lib.sh
[[ $? -ne 0 ]] && exit 1

for i in gfortran libomp; do 
    sudo yum install $i -y ; 
done

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
    
    torchwhl=`find . -name 'rocm_torch*.tar'`
    dirname="torch"
    echo $dirname
    mkdir $dirname ; pushd $dirname
    ln -s ../$torchwhl .
    tar -xvf ./$torchwhl
    pip3 install ./*.whl
    popd
 
    echo $torchwhl

    for i in *tar ; do 
        dirname=`echo $i | awk '{print $1}' FS=. `
        mkdir $dirname ; pushd $dirname
        ln -s ../$i .
        tar -xvf ./$i 
        pip3 install ./*.whl
        popd
    done
popd

conda install mkl-service -y
pip3 install mkl 

tar -xf $LLAMA_PREREQ_PKGS.tar
pwd
ls -l 

pushd $LLAMA_PREREQ_PKGS
mkdir log
bash install.sh 2>&1 | sudo tee log/install.log
popd

sudo ln -s `sudo find /opt -name clang++` /usr/bin/
if [[ -z `which clang++` ]] ; then echo "Error: can not setup or find clang++ in default path" ; exit 1 ; fi

git clone https://bitbucket.org/icl/magma.git
pushd magma

BASHRC=~/.bashrc
BASHRC_EXPORT=./export.md
ROCM_PATH=/opt/rocm/

ls -l $BASHRC

export_bashrc_delim_alt MAGMA_HOME  $MAGMA_HOME
export_bashrc MKLROOT $CONDA_PREFIX
export_bashrc_delim_alt ROCM_PAHT $ROCM_PATH

cp make.inc-examples/make.inc.hip-gcc-mkl make.inc
echo "LIBDIR += -L\$(MKLROOT)/lib" >> make.inc
echo "LIB += -Wl,--enable-new-dtags -Wl,--rpath,\$(ROCM_PATH)/lib -Wl,--rpath,\$(MKLROOT)/lib -Wl,--rpath,\$(MAGMA_HOME)/lib" >> make.inc
echo "DEVCCFLAGS += --amdgpu-target=gfx942" >> make.inc
# build MAGMA
make -f make.gen.hipMAGMA -j
HIPDIR=$ROCM_PATH GPU_TARGET=gfx942 make lib -j 2>&1 | tee make.magma.log
popd

pushd $LLAMA_PREREQ_PKGS

if [[ $SOFT_LINK == 1 ]] ; then
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        ln -s \
        $CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.2 \
        $CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
    done
else
    for i in  libmkl_intel_lp64 libmkl_gnu_thread libmkl_core; do
        rm -rf $CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
        cp \
        $CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.2 \
        $CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/$i.so.1
    done
fi

if [[ -z $CONDA_PREFIX ]] ; then
    echo "Error CONDA_PREFIX is empy. Paths are likely not valid"
    exit 1
fi

# following does not work for python.  even though ldcache includes those paths.
chmod 755 *sh
echo "Use following cmd to run:"
echo 'LD_LIBRARY_PATH=$CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:$MAGMA_HOME/lib ./run_llama2_70b.sh'
popd

echo "$CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib/" | sudo tee /etc/ld.so.conf.d/mkl.conf
echo "$MAGMA_HOME/lib" | sudo tee /etc/ld.so.conf.d/magma.conf
ls -l /etc/ld.so.conf.d/

export_bashrc_delim_alt LD_LIBRARY_PATH $CONDA_PREFIX_1/pkgs/mkl-2023.1.0-h213fc3f_46344/lib:$MAGMA_HOME/lib



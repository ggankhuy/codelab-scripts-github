# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
set -x 

#setup  MINICONDA_SRC_DIR

MINICONDA_SRC_DIR=/home/miniconda3
export MINICONDA_SRC_DIR=$MINICONDA_SRC_DIR

if [[ -z `cat ~/.bashrc | egrep "export.*MINICONDA_SRC_DIR"` ]] ; then
    echo "export MINICONDA_SRC_DIR=$MINICONDA_SRC_DIR" | sudo tee -a ~/.bashrc
fi

LLAMA_PREREQ_PKGS=20240502_quanta_llamav2
CONDA=/$HOME/miniconda3/bin/conda

mkdir -p $MINICONDA_SRC_DIR
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
chmod 755 ./miniconda.sh
bash ./miniconda.sh -b -u -p /$MINICONDA_SRC_DIR
rm -rf ./miniconda.sh

ln -s $MINICONDA_SRC_DIR /$HOME/

# setup CONDA_ENV_NAME

CONDA_ENV_NAME="llama2"
CONDA_ENV_NAME="llama2-test-1"
export CONDA_ENV_NAME=$CONDA_ENV_NAME

if [[ -z `cat ~/.bashrc | egrep "export.*CONDA_ENV_NAME"` ]] ; then
    echo "export CONDA_ENV_NAME=$CONDA_ENV_NAME" | sudo tee -a ~/.bashrc
fi

$CONDA create --name  $CONDA_ENV_NAME python==3.9 -y
$CONDA init
echo "conda activate $CONDA_ENV_NAME" >> ~/.bashrc

# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.

set -x 

function usage()  {
    clear
    echo "$0 --env_name"
    echo "$0 --pkg_name [without file extension tar.gz]"
}

for var in "$@"
do
    echo var: $var
    case "$var" in
        *--help*)
            usage
            exit 0
            ;;
        *--env_name=*)
            p_env_name=`echo $var | awk -F '=' '{print $2}'`
            echo "env_name from cmdline: $p_env_name" 
            ;;
        *--pkg_name=*)
            p_pkg_name=`echo $var | awk -F '=' '{print $2}'`
            echo "pkg_name from cmdline: $p_pkg_name" 
            ;;
        *)
            echo "Unknown cmdline parameter: $var"
            usage
            exit 1
            ;;
    esac
done
yum install sudo tree git wget -y

#setup  MINICONDA_SRC_DIR

MINICONDA_SRC_DIR=/$HOME/miniconda3_src
export MINICONDA_SRC_DIR=$MINICONDA_SRC_DIR

if [[ -z `cat ~/.bashrc | egrep "export.*MINICONDA_SRC_DIR"` ]] ; then
    echo "export MINICONDA_SRC_DIR=$MINICONDA_SRC_DIR" | sudo tee -a ~/.bashrc
fi

if [[ -z $p_pkg_name ]] ; then
    echo "package name is not specified from cmdline. Using default:"
    echo "20240502_quanta_llamav2"
    LLAMA_PREREQ_PKGS=20240502_quanta_llamav2
else
    LLAMA_PREREQ_PKGS=$p_pkg_name
fi

export LLAMA_PREREQ_PKGS=$LLAMA_PREREQ_PKGS
if [[ -z `cat ~\.bashrc | grep LLAMA_PREREQ_PKGS` ]] ; then
    echo "export LLAMA_PREREQ_PKGS=$LLAMA_PREREQ_PKGS" | sudo tee -a ~/.bashrc
fi

echo "package name is set to: $LLAMA_PREREQ_PKGS"

CONDA=/$MINICONDA_SRC_DIR/bin/conda

mkdir -p $MINICONDA_SRC_DIR
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ./miniconda.sh
chmod 755 ./miniconda.sh
bash ./miniconda.sh -b -u -p /$MINICONDA_SRC_DIR
rm -rf ./miniconda.sh

ln -s $MINICONDA_SRC_DIR /$HOME/

# setup CONDA_ENV_NAME

CONDA_ENV_NAME="llama2"

if [[ -z $p_env_name ]] ; then
    echo "conda env_name is not specified from cmdline. Using default env_name:"
    echo "CONDA_ENV_NAME"
else
    CONDA_ENV_NAME=$p_env_name
fi

echo "CONDA_ENV_NAME is set to $CONDA_ENV_NAME"

export CONDA_ENV_NAME=$CONDA_ENV_NAME

if [[ -z `cat ~/.bashrc | egrep "export.*CONDA_ENV_NAME"` ]] ; then
    echo "export CONDA_ENV_NAME=$CONDA_ENV_NAME" | tee -a ~/.bashrc
fi

$CONDA create --name  $CONDA_ENV_NAME python==3.9 -y
$CONDA init

if [[ -z `cat ~/.bashrc | egrep "export.*env_name"` ]] ; then
    echo "export env_name=$CONDA_ENV_NAME" | tee -a ~/.bashrc
fi

echo "conda activate $CONDA_ENV_NAME" | tee -a ~/.bashrc


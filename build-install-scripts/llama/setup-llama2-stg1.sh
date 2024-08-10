# assumes rocm is installed.
# assumes wheel are present in build/ folder: vllm, gradlib, triton, flash-attn.

# changing the actual instalaltion folder to /home/miniconda3 because centos by default alloc-s 
# only 70gb during installation.
# stage1 part will only setup conda environment, nothing specific to llama2 setup except if user specifies
# set the conda environment name using --env_name parameter.
# --pkg_name parameter. 
# for more info on usage, use --help.
# 
# Issues:
# It is not recommended to use this script to maintain multiple conda environment as each setup will 
# defined many env variables in user's bashrc which will confuse and it is not intelligent yet to handle
# multiple environment. 
# Instead if you need creating new conda environment removing all traces of existing one:
# conda env list to display existing conda environment, and remove old one:
# conda env remove -p <path>.
# delete all related parameters from ~/.bashrc.


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

for i in sudo tree git wget gfortran libomp which; do
    yum install -y $i 
done

#setup  MINICONDA_SRC_DIR

MINICONDA_SRC_DIR=/$HOME/miniconda3_src

source ./lib_bash.sh
[[ $? -ne 0 ]] && exit 1

export_bashrc MINICONDA_SRC_DIR $MINICONDA_SRC_DIR
env | grep MINICONDA

if [[ $p_pkg_name ]] ; then
    LLAMA_PREREQ_PKGS=$p_pkg_name
    export_bashrc LLAMA_PREREQ_PKGS $LLAMA_PREREQ_PKGS
    echo "package name is set to: $LLAMA_PREREQ_PKGS"
fi

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

export_bashrc_delim_alt CONDA_ENV_NAME $CONDA_ENV_NAME

$CONDA create --name  $CONDA_ENV_NAME python==3.9 -y
$CONDA init

sed -i "s/export env_name.*/export env_name=${p_env_name}/g" ~/.bashrc

sed -i 's/conda activate.*//g' ~/.bashrc

echo "conda activate $CONDA_ENV_NAME" | tee -a ~/.bashrc



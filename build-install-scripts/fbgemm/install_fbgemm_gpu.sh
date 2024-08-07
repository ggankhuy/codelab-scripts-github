set -x

function usage()  {
    clear
    echo "$0 --rocm-ver"
}

for var in "$@"
do
    echo var: $var
    case "$var" in
        *--help*)
            usage
            exit 0
            ;;
        *--rocm-ver=*)
            p_rocm_ver=`echo $var | awk -F '=' '{print $2}'`
            echo "env_name from cmdline: $p_env_name"
            ;;
        *)
            echo "Unknown cmdline parameter: $var"
            usage
            exit 1
            ;;
    esac
done

if [[ -z $env_name ]] ; then
    echo "env_name is not defined, please run from previously setup conda environment."
    exit 1
fi

if [[ -z $p_rocm_ver ]] ; then
    echo "Rocm version is not defined."
    usage
    exit 1
fi

conda install -n ${env_name} -y \
    hypothesis \
    numpy \
    scikit-build

pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/rocm$p_rocm_ver/
pip install --pre fbgemm-gpu --index-url https://download.pytorch.org/whl/nightly/rocm$p_rocm_ver/


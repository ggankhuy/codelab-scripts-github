set -x
for var in "$@"
do
    echo var: $var
    case "$var" in
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

TAR_DIR=tar
LIB_PATH=`ls -l lib_bash.sh  | awk '{print $NF}'`
[[ -f $LIB_PATH ]] || exit 1
mkdir $TAR_DIR
ls -l lib_bash.sh
[[ -f $pkg_name.tar ]] || exit 1
tar -cvf $pkg_name-rocm.tar $pkg_name.tar $TAR_DIR/llama-rocm setup-llama2-stg1.sh setup-llama2-stg2.sh run-llama2.sh $LIB_PATH setup.md

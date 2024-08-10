set -x
function usage() {
    echo "$0 --pkg_name=<name of tar package containing wheels>"
    exit 1
}
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

[[ $p_pkg_name ]] || usage
TAR_DIR=tar
LIB_PATH=`ls -l lib_bash.sh  | awk '{print $NF}'`
PKG_PATH=`ls -l $p_pkg_name.tar  | awk '{print $NF}'`
[[ -f $LIB_PATH ]] || exit 1
rm -rf $TAR_DIR
mkdir $TAR_DIR
ls -l lib_bash.sh
COMMIT=commit.md

git branch | tee $COMMIT
git log | head -1 | tee -a $COMMIT
COMMIT6=`git log | head -1 | awk '{print $NF}'  | awk '{print substr($0, 0, 6)}'`

[[ -f $p_pkg_name.tar ]] || exit 1
output_filename=${p_pkg_name}_${COMMIT6}.tar.gz
cp -v \
    setup-llama2-stg1.sh \
    setup-llama2-stg2.sh \
    run-llama2.sh \
    $PKG_PATH \
    $LIB_PATH \
    readme.md \
    $COMMIT \
    $TAR_DIR
pushd $TAR_DIR
tar -cvf $output_filename \
    setup-llama2-stg1.sh \
    setup-llama2-stg2.sh \
    run-llama2.sh \
    $p_pkg_name.tar \
    lib_bash.sh \
    readme.md \
    $COMMIT 
popd
tree -fs $TAR_DIR
tar -tf $TAR_DIR/$output_filename | sudo tee $output_filename.log
echo "---- TAR CONTENTS-----"   
cat $output_filename.log

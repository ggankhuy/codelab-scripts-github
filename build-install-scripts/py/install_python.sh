source ../../api/lib.sh
PYTHON_VER=$1

#use default, or better yet, use system version (yet to be implemented).

if [[ -z $PYTHON_VER ]] ; then
    PYTHON_VER=3.9.12
fi
install_python $1
return $?

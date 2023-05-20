p1=$1
mkdir log
if [[ ! -z $p1 ]] ; then
    python3 build-rocm.py --llvmno --fast --vermajor=5.4 --component=$p1  2>&1 | tee log/build.$p1.log
else
    echo "Please specify component..."
fi

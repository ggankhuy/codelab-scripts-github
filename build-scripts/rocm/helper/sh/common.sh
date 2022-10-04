function build_entry () {
    t2=$SECONDS
    if  [[ ! -z $t1 ]] ; then
        echo Build took $((t2-t1)) seconds 2>&1 | tee -a $LOG_SUMMARY
        echo "............................." 2>&1 | tee -a $LOG_SUMMARY
    else
        echo "Did not log previous build" 2>&1 | tee -a $LOG_SUMMARY
    fi
    L_CURR_BUILD=$1
    echo "............................." 2>&1 | tee -a $LOG_SUMMARY
    echo " Building entry: $L_CURR_BUILD" 2>&1 | tee -a $LOG_SUMMARY
    echo "............................." 2>&1 | tee -a $LOG_SUMMARY
    t1=$SECONDS
}

function setup_root_rocm_softlink () {
    rm ~/ROCm
    ln -s $ROCM_SRC_FOLDER  ~/ROCm
    if [[ $? -ne  0 ]] ; then 
        echo "Error during setting up the softlink ~/ROCm"
        ls -l ~
        exit 1
    fi
}

LOG_DIR=/log/rocmbuild/
NPROC=`nproc`
#ROCM_SRC_FOLDER=~/ROCm-$VERSION
ROCM_SRC_FOLDER=`pwd`
export ROCM_SRC_FOLDER=$ROCM_SRC_FOLDER
ROCM_INST_FOLDER=/opt/rocm-$VERSION.$MINOR_VERSION
LOG_SUMMARY=$LOG_DIR/build-summary.log
LOG_SUMMARY_L2=$LOG_DIR/build-summary-l2.log

export LC_ALL=C.UTF-8
export LANG=C.UTF-8

VERSION="5.2"
MINOR_VERSION="0"
mkdir /log/rocmbuild/ -p
ROCM_SRC_FOLDER=/gg/git/ROCm-5.2/

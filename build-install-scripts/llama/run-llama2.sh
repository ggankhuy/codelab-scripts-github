set -x
p1=$1
TOKEN_DATE=`date +%Y%m%d-%H-%M-%S`
[[ -z $p1 ]] && p1=$TOKEN_DATE
LOG_DIR=`pwd`/log/$TOKEN_DATE/
mkdir -p $LOG_DIR
LOG_RUN=$LOG_DIR/$p1.run.log
LOG_DMESG_BEFORE=$LOG_DIR/$p1.dmesg_before.log
LOG_DMESG_AFTER=$LOG_DIR/$p1.dmesg_after.log
LOG_ENV=$LOG_DIR/$p1.environ.log

echo " ----- ROCm ------" | tee $LOG_ENV
cat /opt/rocm/.info/version | tee -a $LOG_ENV
sudo dkms status | tee -a $LOG_ENV
ls -l $LLAMA_PREREQ_PKGS.tar | tee -a $LOG_ENV
echo " ----- pip3 list  ------" | tee -a $LOG_ENV
pip3 list | egrep "torch|triton|gradlib|vllm" | tee -a $LOG_ENV
echo " ----- env variable list ------" | tee -a $LOG_ENV
env | tee -a $LOG_ENV
echo " ----- .bashrc ------" | tee -a $LOG_ENV
cat ~/.bashrc | tee -a $LOG_ENV
echo " ---- $LLAMA_PREREQ_PKGS files ----- " | tee -a $LOG_ENV
tree -fs $LLAMA_PREREQ_PKGS | tee -a $LOG_ENV

pushd $LLAMA_PREREQ_PKGS
dmesg | tee $LOG_DMESG_BEFORE
./run_llama2_70b_bf16.sh 2>&1 | tee $LOG_RUN && echo "---- done ----"   
dmesg | tee $LOG_DMESG_AFTER
popd

tree -fs $LOG_DIR
cat $LOG_DIR/$LOG_RUN

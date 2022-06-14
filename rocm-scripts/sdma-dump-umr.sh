DATE_FIXED=`date +%Y%m%d-%H-%M-%S`
ROCM_ROOT=/opt/rocm-5.2.0
PATH_UMR=/root/gg/git/umr-mod-cp/build/src/app/umr
LOG_FOLDER_BASE=log/umr/$DATE_FIXED
mkdir -p $LOG_FOLDER_BASE
LOG_SUMMARY=$LOG_FOLDER_BASE/summary.log

$PATH_UMR -go 0

for gpu in {0..1}; do
    gpu_addr=`lspci | grep Disp  | head -$((gpu+1)) | tail -1 | cut -d ' ' -f1`
    for sdma in {0..4}; do
        $PATH_UMR --pci 0000:$gpu_addr -RS sdma$sdma | tee $LOG_FOLDER_BASE/umr.gpu-$gpu$i.sdma-$sdma.log
        done
done
        

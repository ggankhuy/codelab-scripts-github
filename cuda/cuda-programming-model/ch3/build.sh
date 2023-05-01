for i in p84 p99 ; do
    nvcc $i.cu ../lib.cu ../kernels.cu -I.. -o $i.out
done

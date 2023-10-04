for i in p84 p99 p108; do
    nvcc $i.cu ../lib.cu ../kernels.cu -I.. -o $i.out
done

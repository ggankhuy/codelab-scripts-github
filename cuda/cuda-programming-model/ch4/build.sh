for i in p182; do
    nvcc $i.cu ../lib.cu ../kernels.cu -I.. -o $i.out
done

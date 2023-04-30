for i in p45 p53; do
    nvcc $i.cu ../lib.cu ../kernels.cu -I.. -o $i.out
done

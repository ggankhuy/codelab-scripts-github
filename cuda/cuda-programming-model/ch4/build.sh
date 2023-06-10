for i in p167 p182; do
    nvcc $i.cu ../lib.cu ../kernels.cu -I.. -o $i.out
done

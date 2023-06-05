for i in p84 p99 ; do
    hipcc $i.cpp ../lib.cpp ../kernels.cpp -I.. -o $i.out
done

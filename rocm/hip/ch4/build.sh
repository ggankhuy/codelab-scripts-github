for i in p182; do
    hipcc $i.cpp ../lib.cpp ../kernels.cpp -I.. -o $i.out
done

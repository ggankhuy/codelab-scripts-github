for i in p45 p53; do
    hipcc $i.cpp ../lib.cpp ../kernels.cpp -I.. -o $i.out
done

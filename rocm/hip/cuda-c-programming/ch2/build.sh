for i in p33 p35 p45 p53; do
    hipcc $i.cpp ../lib.cpp ../kernels.cpp -I.. -o $i.out -lm
done

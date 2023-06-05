for i in p167 p182; do
    hipcc $i.cpp ../lib.cpp ../kernels.cpp -I.. -o $i.out
done

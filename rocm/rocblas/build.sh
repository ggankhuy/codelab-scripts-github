set -x
rm -rf build/*
mkdir build
cd build

for j in helpers.hpp ArgParser.hpp ; do 
    ln -s ../$j .
done

for i in sscal example_sgemm ; do
    ln -s ../$i.cpp
    hipcc --offload-arch=gfx908 --save-temps -c $i.cpp
    hipcc $i.o  /opt/rocm-5.3.0/lib/librocblas.so -o $i.out
done

chmod 755 *.out
cd ..

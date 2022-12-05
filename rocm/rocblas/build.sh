set -x
rm -rf log/*
mkdir log/
rm -rf build/*
mkdir build
cd build

for j in helpers.hpp ArgParser.hpp ; do 
    ln -s ../$j .
done

for i in sscal example_sgemm example_sgemm_strided_batched; do
    ln -s ../$i.cpp
    hipcc --offload-arch=gfx908 --save-temps -c $i.cpp
    hipcc $i.o  /opt/rocm-5.3.0/lib/librocblas.so -o $i.out
    ROCBLAS_LAYER=5 TENSILE_DB=8000 ./$i.out 2>&1 | tee ../log/ROCBLAS_LAYER.5.TENSILE_DB.8000.$i.log
    ROCBLAS_LAYER=4 TENSILE_DB=8000 ./$i.out 2>&1 | tee ../log/ROCBLAS_LAYER.4.TENSILE_DB.8000.$i.log
done

chmod 755 *.out
cd ..

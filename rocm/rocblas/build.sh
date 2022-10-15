ROCM_VERSION=5.2.0
rm -rf build/*
mkdir build
cd build
for FILENAME in sscal example_sgemm; do 
    echo building $FILENAME...
    ln -s ../$FILENAME.cpp
    for i in helpers.hpp ArgParser.hpp ; do 
        ln -s ../$FILENAME .
    done

    hipcc --offload-arch=gfx908 --save-temps -c $FILENAME.cpp
    hipcc $FILENAME.o  /opt/rocm-$ROCM_VERSION/lib/librocblas.so -o $FILENAME.out
done
cd ..

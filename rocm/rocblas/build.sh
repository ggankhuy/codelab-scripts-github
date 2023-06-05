<<<<<<< HEAD
# make sure to install rocblas-devel (yum) or rocblas-dev (apt) package.
set -x
for i in sscal example_sgemm example_sgemm_strided_batched; do
    sudo hipcc -lrocblas  $i.cpp -o $i.out
    ROCBLAS_LAYER=5 TENSILE_DB=8000 ./$i.out 2>&1 | tee ../log/ROCBLAS_LAYER.5.TENSILE_DB.8000.$i.log
    ROCBLAS_LAYER=4 TENSILE_DB=8000 ./$i.out 2>&1 | tee ../log/ROCBLAS_LAYER.4.TENSILE_DB.8000.$i.log
done

chmod 755 *.out
cd ..
=======
set -x
ENABLE_DEBUG_SYMBOLS=1

OPT_DEBUG=""

if [[ $ENABLE_DEBUG_SYMBOLS -eq 1 ]] ; then OPT_DEBUG=" -g " ; fi

for FILENAME in sscal example_sgemm ; do
    for i in helpers.hpp ArgParser.hpp ; do 
        BUILD_DIR=build-$FILENAME
        mkdir $BUILD_DIR
        rm -rf $BUILD_DIR/*
        cd $BUILD_DIR
        #ln -s ../$i .
        ln -s ../$FILENAME.cpp .
        hipcc  --offload-arch=gfx908 --save-temps -c $FILENAME.cpp
        hipcc $FILENAME.o  /opt/rocm/lib/librocblas.so -o $FILENAME.out
        hipcc $OPT_DEBUG --offload-arch=gfx908 --save-temps -c $FILENAME.cpp
        hipcc $FILENAME.o  /opt/rocm/lib/librocblas.so -o $FILENAME.dbg.out
        cd ..
    done
done


>>>>>>> dev-gpu

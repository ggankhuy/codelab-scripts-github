FILENAME=p53
mkdir build/$FILENAME -p  ; cd build/$FILENAME
ln -s ../../$FILENAME.cu .
nvcc $FILENAME.cu ../lib/lib.c -o $FILENAME.out
./$FILENAME.out
cd ../..


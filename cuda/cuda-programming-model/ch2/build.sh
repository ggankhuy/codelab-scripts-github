FILENAME=p45
mkdir build/$FILENAME -p  ; cd build/$FILENAME
ln -s ../../$FILENAME.cu .
nvcc $FILENAME.cu -o $FILENAME.out
./$FILENAME.out
cd ../..


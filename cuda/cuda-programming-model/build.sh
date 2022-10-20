FILENAME=p33
mkdir build ; cd build
ln -s ../$FILENAME.cu .
nvcc $FILENAME.cu -o $FILENAME.out
./$FILENAME.out
cd ..


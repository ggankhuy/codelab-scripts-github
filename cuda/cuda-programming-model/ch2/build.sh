FILENAME=p53
LIBNAME=liblib1
gcc lib1/*.c -c -fPIC
gcc *.o -shared -o $LIBNAME.so
cp $LIBNAME.so /usr/lib/

mkdir build/$FILENAME -p  ; cd build/$FILENAME
ln -s ../../$FILENAME.cu .
nvcc $FILENAME.cu -llib1 -o $FILENAME.out
./$FILENAME.out
cd ../..


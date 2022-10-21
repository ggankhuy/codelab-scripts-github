FILENAME=p53
LIBNAME=liblib1
TEST=test
TEST_MODE=0
DYN_BUILD=0

if [[ $DYN_BUILD -eq 1 ]] ; then
    gcc lib1/*.c -c -fPIC
    gcc *.o -shared -o $LIBNAME.so
    rm -rf /usr/lib/$LIBNAME.so
    cp $LIBNAME.so /usr/lib/
    chmod 755 /usr/lib/$LIBNAME.so

    echo ldconfig:
    ldconfig
    ldconfig -p | grep $LIBNAME
    objdump -T /usr/lib/$LIBNAME.so

    if [[ $TEST_MODE -eq 1 ]] ; then
        echo "test mode..."
        gcc $TEST.c -llib1 -o $TEST.out
    else
        echo "normal mode..."   
        mkdir build/$FILENAME -p  ; cd build/$FILENAME
        ln -s ../../$FILENAME.cu .
        nvcc $FILENAME.cu -L/usr/lib/ -llib1 -o $FILENAME.out
        ./$FILENAME.out
        cd ../..
    fi
else
fi

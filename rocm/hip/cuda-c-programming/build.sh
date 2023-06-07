for FILE in * ; do
    if [[ ` echo $FILE | grep cpp$` ]] ; then
        hipcc $FILE ../lib.cpp ../kernels.cpp -I.. -o $FILE.out
    fi
done

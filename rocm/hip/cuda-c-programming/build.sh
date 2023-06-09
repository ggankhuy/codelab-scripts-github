for FILE in * ; do
    echo "building $FILE..."
    if [[ ` echo $FILE | grep cpp$` ]] ; then
        hipcc $FILE ../lib.cpp ../kernels.cpp -I.. -o $FILE.out
    fi
done

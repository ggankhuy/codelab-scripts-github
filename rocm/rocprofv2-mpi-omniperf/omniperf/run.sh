FILENAME=p61
hipcc $FILENAME.cpp -o $FILENAME.out
omniperf profile -n vcopy_all --  ./$FILENAME.out

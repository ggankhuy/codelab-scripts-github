CONFIG_DEBUG_SYMBOLS=1
FILENAME=p41
if [[ $CONFIG_DEBUG_SYMBOLS -eq 1 ]] ; then OPT_DEBUG=" -g" ; fi

hipcc $OPT_DEBUG $FILENAME.cpp -o $FILENAME.dbg.out
hipcc $FILENAME.cpp -o $FILENAME.out


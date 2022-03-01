./clean.sh
hipcc p41.cpp
rocprof --hip-trace -i input.xml  -d ./prof ./a.out

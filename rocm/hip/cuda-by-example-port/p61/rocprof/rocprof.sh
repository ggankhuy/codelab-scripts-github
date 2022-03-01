hipcc p188.cpp
rocprof --trace-start on ./a.out
cat results.csv | column -t -s,

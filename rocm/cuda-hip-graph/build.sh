#for i in 1 2 3; do 
for i in 3; do 
    sudo hipcc ex-$i.cpp 
    sudo rocprof --hip-trace ./a.out 
    sudo mkdir $i 
    sudo cp results.json $i/ ; done

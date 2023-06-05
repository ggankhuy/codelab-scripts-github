mkdir build
cd build
ln -s ../p41.cpp .
hipcc --save-temps p41.cpp -o p41.out
cd ..

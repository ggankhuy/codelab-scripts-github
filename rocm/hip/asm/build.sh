FILENAME=mm
mkdir build
cd build
ln -s ../$FILENAME.cpp .
hipcc --save-temps $FILENAME.cpp -o $FILENAME.out
cd ..

FILENAME=mm
mkdir -p build/$FILENAME
cd build/$FILENAME
ln -s ../../$FILENAME.cpp .
hipcc --save-temps $FILENAME.cpp -o $FILENAME.out
cd ../..

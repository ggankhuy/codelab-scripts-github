FILENAME=mm
#FILENAME=p41
mkdir -p build/$FILENAME
cd build/$FILENAME
rm -rf ./*
ln -s ../../$FILENAME.cpp .
hipcc --save-temps $FILENAME.cpp -o $FILENAME.out -g -O0
cd ../..

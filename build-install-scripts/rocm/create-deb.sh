DIR_NAME=rocm-builder
mkdir -p $DIR_NAME/usr/local/bin
mkdir $DIR_NAME/DEBIAN
cp -vr *.py *.dat sh $DIR_NAME/usr/local/bin
cp control $DIR_NAME/DEBIAN/
dpkg-deb --build --root-owner-group $DIR_NAME


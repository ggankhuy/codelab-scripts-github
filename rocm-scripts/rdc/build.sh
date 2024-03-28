# this script just combines build-rdc.sh and build-grpc.sh in this same folder.
# those 2 can be removed once this one known to work.

./build-grpc.sh 2>&1 | tee build-grpc.log
./build-rdc.sh 2>&1 | tee build-rdc.log

# this script just combines build-rdc.sh and build-grpc.sh in this same folder.
# those 2 can be removed once this one known to work.
# Currently this build scripts has been tested on following environment only. All other variance, specially
# yum based O/S may fail and/or have unpredictable result.

# Ubuntu:22.04 / ROCm-6.2 13680 MI100.

#./build-grpc.sh 2>&1 | tee build-grpc.log
./build-rdc.sh 2>&1 | tee build-rdc.log

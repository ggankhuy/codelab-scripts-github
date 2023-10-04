FILENAME=p61
if [[ -z `which omniperf` ]] ; then
    echo "Error: omniperf is not found. Make sure it is installed and its path is in PATH env var"
    exit 1
fi
hipcc $FILENAME.cpp -o $FILENAME.out
omniperf profile -n vcopy_all --  ./$FILENAME.out
output_dir=workloads/vcopy_all/mi100/

if [[ ! -d $output_dir ]] ; then
    echo "Error: $output_dir is not found. THis script has been tested on mi100 platform. Adjust the variables as necessary for other GPUs."
    exit 1
fi
omniperf analyze -p $output_dir

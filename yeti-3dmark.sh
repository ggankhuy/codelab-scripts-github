function usage() {
	clear
	echo "Usage: "
	echo "$0 <mode> <option>"
	echo "where <mode>  is either yeti or linux"
	echo "where <options> is either 0 1 2"
	echo "0: - nostream"
	echo "1: - stream with 1 pc"
	echo "2: - stream with 2 pc"
	exit 1
}
	
p1=$1
p2=$2	
mode=0		# 0 for yeti, 1 for linux
option=0

if [[ -z $p1 ]]  || [[ -z $p2 ]]; then
	usage
fi 

if [[ $p1 == "linux" ]] ; then
	echo "linux option is not implemented yet. Sorry."
	exit 1
elif [[ $p1 == "yeti" ]] ; then
	echo "yeti mode is seleted."
	mode=0
else
	echo "invalid mode: $p1. Exiting..."
	exit 1
fi

if [[ $p2 -eq  0 ]] ; then
	echo "no stream option is selected."
	option=0
elif  [[ $p2 -eq 1 ]] ; then
	echo "stream with 1 pc is selected."
	option=1
elif  [[ $p2 -eq 2 ]] ; then
	echo "stream with 2 pc is selected."
	option=2
else
	echo "Invalid option: $p2. Exiting..."
	exit 1
fi

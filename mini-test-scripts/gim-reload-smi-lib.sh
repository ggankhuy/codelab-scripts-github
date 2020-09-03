p1=$1
p2=$2
if [[ -z $p1 ]] ;then p1=3 ; fi
if [[ -z $p2 ]] ;then p2=1 ; fi
lspci -nd 1002:
clear
DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=`pwd`/log/$DATE
mkdir -p $LOG_FOLDER
GPUS=`lspci -nd 1002: | wc -l`
echo "GPUS: $GPUS"
GPU_BDFS=()

for ((i=1; i<=$GPUS; i++)); do
	GPU_BDFS+=(`lspci -nd 1002: | head -$i  | tail -1 | tr -s ' ' | cut -d ' ' -f1`)
done
echo "GPU_BDFS: "

for value in ${GPU_BDFS[@]}
do
     echo $value
done

pushd /usr/src/gim*/smi-lib
make clean ; make
cd examples
cd basic-query
make clean ;make
if [[ $? -ne 0 ]] ; then
	echo "Failed to build  smi-lib utility..."
	ls -l 
	#exit 1
fi
popd
	

echo p1: $p1
for (( i=1 ; i <=$p1 ; i++ )) ; do
	mkdir -p $LOG_FOLDER/$i
	echo "Loading gim $i th time..."
	modprobe gim
	lsmod | grep -i gim
	ret=$?
	echo "Gim load result: $ret"

	dmesg > $LOG_FOLDER/$i/dmesg.gim.load.$DATE.$i.log
	lspci | grep -i amd > $LOG_FOLDER/$i/lspci.gim.load.$DATE.$i.log
	cat /proc/iomem > $LOG_FOLDER/$i/iomem.gim.load$DATE.$i.log

	for j in ${GPU_BDFS[@]} ; do
		lspci -s $j -vvv >> $LOG_FOLDER/$i/lspci.gim.load.$DATE.$i.log
	done
	dmesg --clear

	if [[ $ret -ne 0 ]] ; then
		echo "Error loading gim..."
		exit 1
	fi

	sleep 2

	pushd /usr/src/gim*/smi-lib/examples/basic-query

	if [[ ! -f ./alloc_vf_with_parameters ]] ; then
		echo "Error: ./alloc_vf_with_parameters utility does not exist in current directory or failed to build."
		exit 1
	fi

	./alloc_vf_with_parameters $p1 $p2 | tee -a $LOG_FOLDER/$i/alloc_vf_with_parameters.$DATE.$i.log
	lspci | grep -i amd > $LOG_FOLDER/$i/lspci.alloc.vf.$DATE.$i.log
	for j in ${GPU_BDFS[@]} ; do
		lspci -s $j -vvv >> $LOG_FOLDER/$i/lspci.alloc.vf.$DATE.$i.log
	done
	cat /proc/iomem > $LOG_FOLDER/$i/iomem.alloc.vf.$DATE.$i.log
	dmesg > $LOG_FOLDER/$i/dmesg.alloc.vf.with.parameters.$DATE.$i.log
	dmesg --clear
	popd

	echo "Unload gim $i th time..."
	modprobe -r gim 
	lsmod | grep -i gim
	ret=$?
	echo "Gim unload result: $ret"
	dmesg > $LOG_FOLDER/$i/dmesg.gim.unload.$DATE.$i.log
	dmesg --clear
	lspci | grep -i amd > $LOG_FOLDER/$i/lspci.gim.unload.$DATE.$i.log
	cat /proc/iomem > $LOG_FOLDER/$i/iomem.gim.unload.$DATE.$i.log

	for j in ${GPU_BDFS[@]} ; do
		lspci -s $j -vvv >> $LOG_FOLDER/$i/lspci.gim.unload.$DATE.$i.log
	done

	#if [[ $ret -ne 0 ]] ; then
	#echo Error unloading gim...
	#	exit 1
	#fi
	sleep 2
done

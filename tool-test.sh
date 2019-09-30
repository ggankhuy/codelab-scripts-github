ATITOOL=/root/tools/atitool/tool-test/atitool
DATE=`date +%Y%m%d-%H-%M-%S`

if [[ -z $ATITOOL ]] ; then
	echo $ATITOOL does not exit!
	exit 1
fi

$ATITOOL 
$ATITOOL pm
$ATITOOL  -pmlogall -pmperiod=100 -pmcount=500 -pmoutput=atitool.$DATE.log -pmnoesckey

$ATITOOL  hwid
$ATITOOL -devid 
$ATITOOL -revid
$ATITOOL -subsysvendorid
$ATITOOL -subsysid
$ATITOOL -cu
$ATITOOL -biosbuildnum

$ATITOOL -ppdpmstatus
$ATITOOL -ppstatus
$ATITOOL -pplist
$ATITOOL -pplist=full

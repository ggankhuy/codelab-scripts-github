# 	Runs a series of non-interactive test runs for agt, amdvbflash and atitool.
# 	It does not check for appropraite tool version, instead merely calls each
#	command based on 3 variables defined below that points to path: 
#	ATITOOL, AGTTOOL and AMDVBFLASH.
#	Tester is responsible for placing appropriate tools under the paths defined.

ATITOOL=/root/tools/atitool/tool-test/atitool
AGTTOOL=/root/tools/agt/tool-test/agt
AMDVBFLASH=/root/tools/amdbvflash/tool-test/amdvbflash
AMDVBFLASH=/root/tools/amdvbflash/tool-test/amdvbflash
DATE=`date +%Y%m%d-%H-%M-%S`

echo =======================================
echo STARTING ATITOOL test
echo =======================================
if [[ -z $ATITOOL ]] ; then
	echo $ATITOOL does not exit!
	exit 1
fi

$ATITOOL 
$ATITOOL pm
$ATITOOL  -pmlogall -pmperiod=100 -pmcount=500 -pmoutput=atitool.$DATE.log 

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

echo =======================================
echo STARTING AGT TOOL test
echo =======================================

$AGTTOOL clock
$AGTTOOL -clkstatus 
$AGTTOOL  -pmlogall -pmperiod=100 -pmcount=500 -pmoutput=agttool.$DATE.log
$AGTTOOL -ppdpmstatus
$AGTTOOL -ppstatus
$AGTTOOL -pplist
$AGTTOOL -pplist=full

echo =======================================
echo STARTING AMDVBFLASH TOOL test
echo =======================================
$AMDVBFLASH -ai
$AMDVBFLASH -s 0 amdvblfash.out.vbios.$DATE.bin



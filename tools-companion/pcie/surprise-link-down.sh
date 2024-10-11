# trigger surprise link down using pcie config register:
# 1 cap pointer in std space: 0x34 (offset to 1st capabilities)
# 2 walk through capabilities.
# 3 find pcie express capability register (capID: 10h)
# 4 offset 10h/word size rx in pcie link capability.
# 5 set bit 4.

# warning: this wil make the device fall of pcie tree and make sure to use 
# it carefully.

# usage:
# ./surprise-link-down.sh bug dev fcn  (decimal)
# example:
# ./surprise-link-down.sh 0 08 03 (cause SLD in bus 0, device 08, function 03)
# ./surprise-link-down.sh 0 12 03 (cause SLD in bus 0, device 12 (0xc), function 03)

BUS=$1
DEV=$2
FCN=$3

set -x 
[[ ! -z $BUS ]] || exit 1 
[[ ! -z $DEV ]] || exit 1 
[[ ! -z $FCN ]] || exit 1 

RX_STD_CAP_PT=`lspci -s $BUS:$DEV.$FCN -x | grep 30: | awk '{print $6}'`
RX_STD_CAP_PT=`echo "ibase=16; $RX_STD_CAP_PT" | bc`
[[ -z $RX_STD_CAP_PT ]] && exit 1
capId=""
RX_STD_CAP_PT_ROW=`echo $((RX_STD_CAP_PT & 0xf0))`
RX_STD_CAP_PT_ROW_16=`echo "obase=16; $RX_STD_CAP_PT_ROW" | bc`
RX_STD_CAP_PT_COL=`echo $((RX_STD_CAP_PT & 0x0f))`
RX_STD_CAP_PT_COL_16=`echo "obase=16; $RX_STD_CAP_PT_COL" | bc`
currCapId=$capId
currCapPtr=$RX_STD_CAP_PT
currCapPtrRow_16=$RX_STD_CAP_PT_ROW_16
currCapPtrCol_16=$RX_STD_CAP_PT_COL_16
counter=0
while [[ $currCapId != "10" ]] && [[ $counter -lt 4 ]] && [[ $currCapPtr -ne "00" ]] ; 
do  
    echo "------------ loop $counter ---------"

    # process capID before pointing to next cap.

    CAP_ID_1=`sudo lspci -s $BUS:$DEV.$FCN -xxx | grep $currCapPtrRow_16: | head -1`
    CAP_ID_2=`echo $CAP_ID_1 | awk -v a=$((currCapPtrCol_16+2)) '{print $a}'`
    currCapId=$CAP_ID_2
    if [[ $currCapId == "10" ]] ; then break; fi

    # point to next cap.

    NEXT_CAP_PTR_1=`sudo lspci -s $BUS:$DEV.$FCN -xxx | grep $currCapPtrRow_16: | head -1`
    NEXT_CAP_PTR_2=`echo $NEXT_CAP_PTR_1 | awk -v b=$((currCapPtrCol_16+3)) '{print $b}'`

    currCapPtr=$NEXT_CAP_PTR_2
    currCapPtr=`echo "ibase=16; $currCapPtr" | bc`
    currCapPtrRow=`echo $((currCapPtr & 0xf0))`
    currCapPtrRow_16=`echo "obase=16; $currCapPtrRow" | bc`
    currCapPtrCol=`echo $((currCapPtr & 0x0f))`
    currCapPtrCol_16=`echo "obase=16; $currCapPtrCol" | bc`
    counter=$((counter+1))
done 

echo "currCapID: $currCapId"
echo "currCapPtr: $currCapPtr"

# linkControl is word at 0x10. But since we are setting bit4, due to big indian
# we are reading 11th byte.

echo rxPcieCapLinkCtrl_b1=$((currCapPtr+0x11))
echo rxPcieCapLinkCtrl_b2=$((currCapPtr+0x10))
echo rxPcieCapLinkCtrl=$((currCapPtr+0x10))

# Reading is not necessary just for test purposes.

rxPcieCapLinkCtrlRow=`echo $((rxPcieCapLinkCtrl & 0xf0))`
rxPcieCapLinkCtrlRow_16=`echo "obase=16; $rxPcieCapLinkCtrlRow" | bc`
rxPcieCapLinkCtrlCol=`echo $((rxPcieCapLinkCtrl & 0x0f))`
rxPcieCapLinkCtrlCol_16=`echo "obase=16; $rxPcieCapLinkCtrlCol" | bc`
rxPcieCapLinkCtrlVal=`sudo lspci -s $BUS:$DEV.$FCN -xxx | grep $rxPcieCapLinkCtrlVal_16: | head -1`
rxPcieCapLinkCtrlVal=`echo $CAP_ID_1 | awk -v a=$((rxPcieCapLinkCtrlVal_16+2)) '{print $a}'`

# set the link disable bit 4.

setpci -s $BUS:$DEV.$FCN $rxPcieCapLinkCtrl.w=0x10
exit 0

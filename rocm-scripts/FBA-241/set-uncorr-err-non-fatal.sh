#!/bin/bash
#
# Disable ACS on every device that supports it
#
PLATFORM=$(dmidecode --string system-product-name)
logger "PLATFORM=${PLATFORM}"
# Enforce platform check here.
#case "${PLATFORM}" in
        #"OAM"*)
                #logger "INFO: Disabling ACS is no longer necessary for ${PLATFORM}"
                #exit 0
                #;;
        #*)
                #;;
#esac
# must be root to access extended PCI config space
if [ "$EUID" -ne 0 ]; then
        echo "ERROR: $0 must be run as root"
        exit 1
fi
for BDF in `lspci -d "*:*:*" | awk '{print $1}'`; do
        # skip if it doesn't support ACS
        skip=0;
        aer_cap=`lspci -s ${BDF} | grep -i "Advanced Error Reporting"`

        if [ $aer_cap -eq "" ]; then
                #echo "${BDF} does not support ACS, skipping"
                skip=1;
        fi
        if [ $skip != 1 ]; then
          logger "Disabling ACS on `lspci -s ${BDF}`"
          setpci -v -s ${BDF} ECAP01+C.L=00000000
          #setpci -v -s ${BDF} ECAP_ACS+0x6.w=0000
          if [ $? -ne 0 ]; then
                logger "Error disabling ACS on ${BDF}"
                continue
          fi
          #NEW_VAL=`setpci -v -s ${BDF} ECAP01+C.L | awk '{print $NF}'`
          lspci -s $i -vvv| grep -i UESvrt
          #NEW_VAL=`setpci -v -s ${BDF} ECAP_ACS+0x6.w | awk '{print $NF}'`
          #if [ "${NEW_VAL}" != "0000" ]; then
          #      logger "Failed to disable ACS on ${BDF}"
          #      continue
          #fi
        fi
done
exit 0

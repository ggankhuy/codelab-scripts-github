#!/bin/bash
#
# Disable ACS on every device that supports it
#
PLATFORM=$(dmidecode --string system-product-name)
DEBUG=0
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
        dev_exist=`lspci -s ${BDF}`

        if [[ -z dev_exist ]] ; then skip=1 ; fi
        echo ---
        if [[ $DEBUG -eq 1 ]] ; then  echo ${BDF} ; fi
        aer_cap=`lspci -s ${BDF} -vvv | grep -i "Advanced Error Reporting"`

        if [[ -z $aer_cap ]]; then
                echo "${BDF} does not support AER, skipping"
                skip=1;
        fi

        if [ $skip != 1 ]; then
          echo "Readback of UESVrt on ${BDF}: "
          lspci -s ${BDF} -vvv| grep -i UESvrt
          echo "Setting all uncorr err as non-fatal on `lspci -s ${BDF}`"
          setpci -v -s ${BDF} ECAP01+C.L=00000000

          if [ $? -ne 0 ]; then
                echo "!Error setting AER setting on ${BDF}"
                continue
          fi
          echo "Readback of UESVrt on ${BDF}: "
          lspci -s ${BDF} -vvv| grep -i UESvrt
        fi
done
exit 0

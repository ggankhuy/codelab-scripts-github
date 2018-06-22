#!/bin/bash
#
# Disable ACS on every device that supports it
#
PLATFORM=$(dmidecode --string system-product-name)
DEBUG=0

# Controls disabling SERR/PERR on command register (04h)
CONFIG_DISABLE_SERR_PERR=1

# Control disabling the fatal/non-fatal/corr err on device control register. (cap.exp.08h)
CONFIG_DISABLE_AER_DEVCTL=1

# Control disabling SERR/PERR on bridge control (SERR propagation from secondary to primary side) (3eh)
# type 1 configuration, it applicable to bridge devices only.

CONFIG_DISABLE_AER_BRIDGE=1

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

echo "Final status: all AER/SERR/PERR Rx-s:"
lspci -vvv | egrep "PERR|SERR|[0-9a-f][0-9a-f]:[0-9a-f]|NonFatalErr.*FatalErr"

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

# CLEAR THE BIT  

echo ----
echo clear the bit...

A=`setpci -s 2f:00.0 04.L`

echo A1: $A
A=$(( 16#$A ))
echo A2: $A
B=$((A & 16#FFFFFFFE))
echo B1: $B
B=`printf '%x\n' $B`
echo B2: $B

# SET THE BIT

echo ----
echo clear the bit...

A=`setpci -s 2f:00.0 04.L`

echo A1: $A
A=$(( 16#$A ))
echo A2: $A
B=$((A | 16#FFFF))
echo B1: $B
B=`printf '%x\n' $B`
echo B2: $B

          if [[ CONFIG_DISABLE_SERR_PERR -eq 1 ]] ; then
              A=`lspci -s ${BDF} -vvv | grep Control:`
              if [[ -z $A ]] ; then
                  echo "${BDF}: Bypassing SERR disable as I am not finding DevCtrl in PCIe CAP space."
              else
                  A=`setpci -v -s ${BDF} 04.L`
                  A=$(( 16#$A ))
                  B=$((A & 16#FFFFFEFF))
                  B=`printf '%x\n' $B`
                  setpci -v -s ${BDF} 04.L=$B
              fi
            
          else
              echo "Bypassing disable of SERR/PERR on ${BDF}
          fi

          if [[ CONFIG_DISABLE_AER_DEVCTL -eq 1 ]] ; then
              A=`lspci -s ${BDF} -vvv | grep DevCtl:`
              if [[ -z $A ]] ; then
                  echo "${BDF}: Bypassing SERR disable as I am not finding DevCtrl in PCIe CAP space."
              else
                  A=`setpci -v -s ${BDF} CAP_EXP+08.L`
                  A=$(( 16#$A ))
                  B=$((A & 16#FFFFFEFF))
                  B=`printf '%x\n' $B`
                  setpci -v -s ${BDF} CAP_EXP+08.L=$B
              fi
          else
              echo "Bypassing disable of fatal/non-fatal/corr DevCtl on ${BDF}
          fi

          if [[ CONFIG_DISABLE_AER_BRIDGE -eq 1 ]] ; then
              A=`lspci -s ${BDF} -vvv | grep BridgeCtl:`

              if [[ -z $A ]] ; then
                  echo "${BDF}: Bypassing SERR disable as not a Type 1.bridge device"
              else
                  A=`setpci -v -s ${BDF} 3E.L`
                  A=$(( 16#$A ))
                  B=$((A & 16#FFFFFEFF))
                  B=`printf '%x\n' $B`
                  setpci -v -s ${BDF} 3E.L=$B
              fi

          else
              echo "Bypassing SERR disable on /Type 1/ ${BDF}
          fi

          if [ $? -ne 0 ]; then
                echo "!Error setting AER setting on ${BDF}"
                continue
          fi
          echo "Readback of UESVrt on ${BDF}: "
          lspci -s ${BDF} -vvv| grep -i UESvrt
        fi
done
exit 0

echo "Final status: all AER/SERR/PERR Rx-s:"
lspci -vvv | egrep "PERR|SERR|[0-9a-f][0-9a-f]:[0-9a-f]|NonFatalErr.*FatalErr"

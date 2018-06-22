#!/bin/bash
#
# Disable ACS on every device that supports it
#
PLATFORM=$(dmidecode --string system-product-name)
DEBUG=0

# Controls disabling SERR/PERR on command register (04h)
CONFIG_DISABLE_SERR_PERR=1
DATE=`date +%Y%m%d-%H-%M-%S`
LOG_FOLDER=./log/$DATE
LOG_L1=$LOG_FOLDER/$DATE-set-uncorr-err-non-fatal.log
LOG_L2=$LOG_FOLDER/$DATE-set-uncorr-err-non-fatal-L2.log
LOG_INIT=$LOG_FOLDER/$DATE-init.log
LOG_FINAL=$LOG_FOLDER/$DATE-final.log
mkdir -p $LOG_FOLDER

# Control disabling the fatal/non-fatal/corr err on device control register. (cap.exp.08h)
CONFIG_DISABLE_AER_DEVCTL=1

# Control disabling SERR/PERR on bridge control (SERR propagation from secondary to primary side) (3eh)
# type 1 configuration, it applicable to bridge devices only.

CONFIG_DISABLE_AER_BRIDGE=1

echo "init status: all AER/SERR/PERR Rx-s:" | tee $LOG_INIT
lspci -vvv | egrep "PERR|SERR|[0-9a-f][0-9a-f]:[0-9a-f]|NonFatalErr.*FatalErr" | tee -a $LOG_INIT

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
        echo --- | tee -a $LOG_L1 | tee -a $LOG_L2
        if [[ $DEBUG -eq 1 ]] ; then  echo ${BDF} ; fi
        aer_cap=`lspci -s ${BDF} -vvv | grep -i "Advanced Error Reporting"`

        if [[ -z $aer_cap ]]; then
                echo "${BDF}: does not support AER capability, bypassing." | tee -a $LOG_L1
                skip=1;
        fi

        if [ $skip != 1 ]; then
          echo "${BDF}: initread of UESVrt." | tee -a $LOG_L1
          lspci -s ${BDF} -vvv| grep -i UESvrt
          echo "Setting all uncorr err as non-fatal on `lspci -s ${BDF}`" | tee -a $LOG_L1
          setpci -v -s ${BDF} ECAP01+C.L=00000000

          if [ $? -ne 0 ]; then
                echo "${BDF} !Error setting AER setting." | tee -a $LOG_L1
                continue
          fi
          echo "${BDF}: Readback of UEsvrt: " | tee -a $LOG_L1
          lspci -s ${BDF} -vvv| grep -i UESvrt

          if [[ CONFIG_DISABLE_SERR_PERR -eq 1 ]] ; then
              A=`lspci -s ${BDF} -vvv | grep Control:`
              if [[ -z $A ]] ; then
                  echo "${BDF}.Control.0x04: Bypassing SERR disable as I am not finding DevCtrl in PCIe CAP space." | tee -a $LOG_L1 | tee -a $LOG_L2
              else
                  A=`setpci -v -s ${BDF} 04.L`
                  A=$(( 16#$A ))
                  B=$((A & 16#FFFFFEFF))
                  B=`printf '%x\n' $B`
                  setpci -v -s ${BDF} 04.L=$B
              fi
            
          else
              echo "${BDF}.Control.0x04: Bypassing disable of SERR/PERR."
          fi

          if [[ CONFIG_DISABLE_AER_DEVCTL -eq 1 ]] ; then
              A=`lspci -s ${BDF} -vvv | grep DevCtl:`
              if [[ -z $A ]] ; then
                  echo "${BDF}.DevCtl.CAP_EXP0x08: Bypassing SERR disable as I am not finding DevCtrl in PCIe CAP space." | tee -a $LOG_L1 | tee -a $LOG_L2
              else
                  A=`setpci -v -s ${BDF} CAP_EXP+08.L`
                  A=$(( 16#$A ))
                  B=$((A & 16#FFFFFEFF))
                  B=`printf '%x\n' $B`
                  setpci -v -s ${BDF} CAP_EXP+08.L=$B
              fi
          else
              echo "${BDF}.DevCtl.CAP_EXP0x08: Bypassing disable of fatal/non-fatal/corr." | tee -a $LOG_L1 | tee -a $LOG_L2
          fi

          if [[ CONFIG_DISABLE_AER_BRIDGE -eq 1 ]] ; then
              A=`lspci -s ${BDF} -vvv | grep BridgeCtl:`

              if [[ -z $A ]] ; then
                  echo "${BDF}.T1.BridgeCtl.0x3e: Bypassing SERR disable as not a Type 1/bridge device." | tee -a $LOG_L1 | tee -a $LOG_L2
              else
                  A=`setpci -v -s ${BDF} 3E.L`
                  A=$(( 16#$A ))
                  B=$((A & 16#FFFFFEFF))
                  B=`printf '%x\n' $B`
                  setpci -v -s ${BDF} 3E.L=$B
              fi

          else
              echo "${BDF}.T1.BridgeCtl.0x3e: Bypassing SERR disable." | tee -a $LOG_L1 | tee -a $LOG_L2
          fi
        fi
done

echo "final status: all AER/SERR/PERR Rx-s:" | tee $LOG_FINAL
lspci -vvv | egrep "PERR|SERR|[0-9a-f][0-9a-f]:[0-9a-f]|NonFatalErr.*FatalErr" | tee -a $LOG_FINAL

# not working
PW='\#3paleHorse5\#'
USER=master

# working ok.
PW=amd1234
USER=root

CONFIG_IP_GUEST=10.216.54.96
LOOP_COUNT=3
SINGLE_BAR="-----------------------------"
for loop in {1..3} ; do
    echo $SINGLE_BAR
    echo "Current loop: $loop........"

    for cmd in "mkdir fba-212" "sudo dmesg --clear"  "sudo modprobe amdgpu" \
                "sudo dmesg | sudo tee fba-212/dmesg.after.modprobe.amdgpu.'$loop'.log" \
                "sudo dmesg --clear" "sudo rmmod amdgpu" "sudo dmesg | sudo tee fba-212/dmesg.after.rmmod.'$loop'.log"\
        ;do
        echo $cmd
        sshpass -p $PW ssh -o StrictHostKeyChecking=no $USER@$CONFIG_IP_GUEST $cmd
        ret=$?

        if [[ $ret != 0 && $ret != 1 ]] ; then 
            echo "code: $ret. Error executing $cmd on remote $CONFIG_IP_GUEST with credentials: $USER/$PW"
            exit 1
        fi
        sleep 1
    done
done


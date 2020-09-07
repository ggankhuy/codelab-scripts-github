ARR_VM_IP=()
ARR_VM_NO=()
ARR_VM_NAME=()
ARR_VM_VF=()
ARR_VM_PF=()

#   Get pcie address (bdf) of a VM.
#   This function needs to be called with ARR_VM_NAME is filled with running VM-s otherwise result is invalid.
#   input: None.

function get_bdf()
{


    for w in ${ARR_VM_NAME[@]} ; do
        bus=`virsh dumpxml $w | grep "<\hostdev\>" -A 20 | grep hostdev -B 20 | grep -i "address domain" | tr -s ' '  | cut -d ' ' -f4 | tr -s ' ' | cut -d "'" -f2 | cut -d 'x' -f2`
        dev=`virsh dumpxml $w | grep "<\hostdev\>" -A 20 | grep hostdev -B 20 | grep -i "address domain" | tr -s ' '  | cut -d ' ' -f5 | tr -s ' ' | cut -d "'" -f2 | cut -d 'x' -f2`
        fcn=`virsh dumpxml $w | grep "<\hostdev\>" -A 20 | grep hostdev -B 20 | grep -i "address domain" | tr -s ' '  | cut -d ' ' -f6 | tr -s ' ' | cut -d "'" -f2 | cut -d 'x' -f2`

        vf=$bus:$dev.$fcn
        ARR_VM_VF+=( $vf )
        pf=`lspci | grep -i $vf -B 1 | head -1 | cut -d ' ' -f1`
        ARR_VM_PF+=( $pf )

        if [[ $DEBUG -eq 1 ]] ; then echo "PF/VF obtained for VM: ${VM_NAME[$i]}: $pf/$vf" ; fi ;
    done
}

#   Wait untill specified VM becomes pingable.
#   input -
#   $1 - VM_NAME to wait for until it becomes pingable.

function wait_till_ip_read()
{
    p=$1
    echo waiting for $p to become pingable...
    for s in {0..50} ; do
        echo issuing virsh domifaddr $r
        tmpIp=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`

        ping -q -c 4  $tmpIp
        stat=$?

        if [[ $stat -eq 0 ]] ; then
            echo "Can ping $tmpIp now..."
            break
        fi
        sleep 30
    done

    if [[ $stat -ne 0 ]] ; then
        echo "Error: Can not ping $tmpIp for long time with 10 retries..."
        exit 1
    fi

}

#   Wait untill all VMs running VM becomes pingable.
#   For this function, ARR_VM_NAME must be populated properly.
#   input: None.

function wait_till_ips_read()
{
    for r in ${ARR_VM_NAME[@]}
    do
        wait_till_ip_read $r
    done
}

#   Prints all arrays. All arrays must populated properly prior to calling this function.
#   input: None.

#   Prints all arrays. All arrays must populated properly prior to calling this function.
#   input: None.

function print_arrs()
{
    echo $SINGLE_BAR
    echo ${ARR_VM_IP[@]}
    echo ${ARR_VM_NO[@]}
    echo ${ARR_VM_PF[@]}
    echo ${ARR_VM_VF[@]}
    for o in ${ARR_VM_NAME[@]} ; do echo $o; done;
    echo $SINGLE_BAR
}

#   Clear all arrays. Mostly needed since ARR_VM_NO which holds the VM index changes after reboot or power recycle.
#   input:

function clear_arrs()
{
    ARR_VM_IP=()
    ARR_VM_NO=()
    ARR_VM_NAME=()
    ARR_VM_VF=()
    ARR_VM_PF=()
}

#   Populates the arrays ARR_VM_IP, ARR_VM_NO, ARR_VM_NAME respectively.
#   input: 0-based nth VM number. Note that this is not the VM index shows in virsh list.
#   All VM-s must be running prior to calling this function.
#   input: None.

function get_vm_info()
{
    indexNo=$1
    GPU_INDEX=$indexNo
    VM_INDEX=$(($indexNo+1))
    echo "get_vm_info: p1: $1. VM_INDEX: $VM_INDEX..."
    sleep 3
    VM_NAME=`virsh list  | grep gpu | head -$(($VM_INDEX)) | tail -1  | tr -s ' ' | cut -d ' ' -f3`
    VM_NO=`virsh list  | grep gpu | head -$(($VM_INDEX)) | tail -1  | tr -s ' ' | cut -d ' ' -f2`
    VM_IP=`virsh domifaddr $VM_NAME | grep ipv4 | tr -s ' ' | cut -d ' ' -f5 | cut -d '/' -f1`
    ARR_VM_IP+=( $VM_IP )
    ARR_VM_NO+=( $VM_NO )
    ARR_VM_NAME+=( $VM_NAME )
    DMESG_FILE_NAME=/tmp/dmesg-loop-$loopNo-vm-$vmNo.log
    wait_till_ip_read $VM_NAME
}


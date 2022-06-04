#!/bin/bash

source basic-functions.sh
test_case_name="attack-sysfs"
sys_node="/sys/bus/pci/devices/0000"
declare -a pids
gpuvs_path="/sys/bus/pci/drivers/gim"
test_pf_dev_lst=`lspci -n | grep 1002 | grep -E "6864|6860" | awk '{print $1}'`
test_vf_dev_lst=`lspci -n | grep 1002 | grep -E "686c" |awk '{print $1}'`
function stop_all_test()
{
	pids=$1
	attack_sysfs_test_time_in_seconds_float=$(echo "$option_attack_sysfs_time_in_hours * 3600" | bc )
	attack_sysfs_test_time_in_seconds=`echo ${attack_sysfs_test_time_in_seconds_float%.*}`
	attack_sysfs_time=$(date +%s)
	attack_sysfs_start_time=$attack_sysfs_time
	attack_sysfs_test_time_in_seconds=$[ $attack_sysfs_time + $attack_sysfs_test_time_in_seconds ]

	while [ $attack_sysfs_time -lt $attack_sysfs_test_time_in_seconds ];
	do
		attack_sysfs_time=$(date +%s)
		time_running=$[ $attack_sysfs_time - $attack_sysfs_start_time ]
		time_left=$(($attack_sysfs_test_time_in_seconds - $attack_sysfs_time))
		convert_time_from_seconds $time_running
		time_running_str=$time_converted
		convert_time_from_seconds $time_left
		time_left_str=$time_converted
		sleep 1
		printf "........ attack sysfs has run for $time_running_str.............$time_left_str left........ \r"
 		r1=`dmesg | grep "Call Trace"`
                r2=`dmesg | grep "gim error" | grep -vi "Invalid Parameters"`
		if [ "$r1" != "" -o "$r2" != ""  ]; then
                        g_host_status=$g_host_status" there is $r1$r2$r3 in host dmesg"
			dmesg_status=false
			break;
                fi
          
	done
	for pid in $pids
	do
		kill -9 $pid
	done
	if [ "$dmesg_status" = "false" ]; then
		test_fail_exit "$g_host_status" 
	fi
}
function test_fail_exit()
{
	fail_reason=$1
	echo | tee -a $test_report_folder/$report_host_file_name
	add_test_end_time_to_report
       	echo test result: Fail | tee -a $test_report_folder/$report_host_file_name
       	echo "failed reason: $1" | tee -a $test_report_folder/$report_host_file_name
	add_driver_info_to_report
	show_test_report
	exit 1
}
function gim_load_and_reload()
{
	while true
	do
		echo LOAD_GIM 
		modprobe gim
		echo unload gim
		modprobe -r gim
	done		
}
function check_sysfs_gim_info()
{
		echo "cat /sys/bus/pci/drivers/gim/event_log"
		cat /sys/bus/pci/drivers/gim/event_log | tee -a $test_report_folder/sysfs_info.log 
		echo  "cat /sys/bus/pci/drivers/gim/event_bind"
		cat /sys/bus/pci/drivers/gim/event_bind | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/gpubios"
		cat /sys/bus/pci/drivers/gim/gpubios | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/gpuinfo"
		cat /sys/bus/pci/drivers/gim/gpuinfo | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/gpuvs"
		cat /sys/bus/pci/drivers/gim/gpuvs | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/guard"
		cat /sys/bus/pci/drivers/gim/guard | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/guard_status"
		cat /sys/bus/pci/drivers/gim/guard_status | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/guard_threshold"
		cat /sys/bus/pci/drivers/gim/guard_threshold | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/sriov"
		cat /sys/bus/pci/drivers/gim/sriov | tee -a $test_report_folder/sysfs_info.log
		echo "cat /sys/bus/pci/drivers/gim/test_error_log"
		cat /sys/bus/pci/drivers/gim/test_error_log | tee -a $test_report_folder/sysfs_info.log

		for i in ${test_vf_dev_lst[*]}
		do
			echo "cat $sys_node:$i/gpuvf"
			cat $sys_node:$i/gpuvf | tee -a $test_report_folder/sysfs_info.log
		done
}

function run_get_vf()
{
	for i in ${test_pf_dev_lst[*]}
	do
        echo "GG: iter: i: $i, sys_node: $sys_node"
		echo "cat $sys_node:$i/gpuvs"
		cat $sys_node:$i/gpuvs | tee -a $test_report_folder/sysfs_info.log
		echo "echo 16 4096 8 > $sys_node:$i/getvf"
		echo 16 4096 8 > $sys_node:$i/getvf | tee -a $test_report_folder/sysfs_info.log
		echo "echo `cat $sys_node:$i/getvf` > $sys_node:$i/relvf"
		echo `cat $sys_node:$i/getvf` > $sys_node:$i/relvf | tee -a $test_report_folder/sysfs_info.log
		echo "cat $sys_node:$i/relvf"
		cat $sys_node:$i/relvf  | tee -a $test_report_folder/sysfs_info.log
	done
}

function run_sysfs_gim_info()
{
	while true
	do
		check_sysfs_gim_info
	done
}
function run_clear_vf_fb()
{
	for i in ${test_vf_dev_lst[*]}
	do
        echo "GG: issuing clear vf fb for $i..."
		echo "echo 1 > $sys_node:$i/clrvffb"
		echo 1 > $sys_node:$i/clrvffb | tee -a $test_report_folder/gim_info.log
		sleep 1
	done
}
function run_hot_reset()
{
	for i in ${test_pf_dev_lst[*]}
	do	
        echo "GG: issuing hot reset for $i..."
		echo "echo 1 > $sys_node:$i/hot_reset"
		echo 1 > $sys_node:$i/hot_reset | tee -a $test_report_folder/sysfs_info.log
		sleep 1
	done
}

function test_attack_sysfs()
{ 
	echo  | tee -a $test_report_folder/$report_host_file_name
	echo  | tee -a $test_report_folder/$report_host_file_name
	echo  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ | tee -a $test_report_folder/$report_host_file_name
	echo test case: $test_case_name | tee -a $test_report_folder/$report_host_file_name
  
	echo  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ | tee -a $test_report_folder/$report_host_file_name
	echo  | tee -a $test_report_folder/$report_host_file_name
	echo test steps: | tee -a $test_report_folder/$report_host_file_name
	echo "1.load gim" | tee -a $test_report_folder/$report_host_file_name
	echo "2.check sysfs nodes: all of the '/sys/bus/pci/drivers/gim' info for the GIM driver" | tee -a $test_report_folder/$report_host_file_name
	echo "     'event_log', 'event_bind', 'gpubios', 'gpuinfo', 'gpuvs', 'guard', " | tee -a $test_report_folder/$report_host_file_name
	echo "     'guard_status', 'guard_threshold','Sriov', 'test_error_log'" | tee -a $test_report_folder/$report_host_file_name
	echo "      gpuvs info for each PF device,getvf,relvf" | tee -a $test_report_folder/$report_host_file_name
	echo "      gpuvf info for each VF device" | tee -a $test_report_folder/$report_host_file_name
	echo "3.unload gim" | tee -a $test_report_folder/$report_host_file_name
	echo "4.run three threads  simultaneously" | tee -a $test_report_folder/$report_host_file_name
	echo "     a.run gim load and unload" | tee -a $test_report_folder/$report_host_file_name
	echo "     b.cat gim pci info" | tee -a $test_report_folder/$report_host_file_name
	echo "     c.trigger whole gpu reset for PF device" | tee -a $test_report_folder/$report_host_file_name
	echo "       trigger clrvffb for each VF device "| tee -a $test_report_folder/$report_host_file_name
	echo "5, after the loop, to create vm and assign vf to vm"
	echo "6, load guest KMD in all VMs"
	echo "expected result: " | tee -a $test_report_folder/$report_host_file_name
	echo  "guest KMD can load successfully" | tee -a $test_report_folder/$report_host_file_name
	echo | tee -a $test_report_folder/$report_host_file_name
	echo test case option:| tee -a $test_report_folder/$report_host_file_name
	echo option_attack_sysfs_time_in_hours=$option_attack_sysfs_time_in_hours | tee -a $test_report_folder/$report_host_file_name
	echo operation_mode=$operation_mode | tee -a $test_report_folder/$report_host_file_name

	test_begin_time_in_seconds=`date +%s`
	echo begin...
	echo
	sleep 2
 
	vf_num_per_gpu=$option_gim_vf_num
	#echo load GIM
	#total_vm_to_run=$(( $vf_num_per_gpu * $total_gpu_num_support_sriov))
	#LOAD_GIM 
	sleep 1
	#echo start to check info
	#check_sysfs_gim_info 
	#echo UNLOAD_GIM
	#UNLOAD_GIM 1
	#echo "start to run whole reset and get sysfs info simultaneously"
	#run_hot_reset &
	#pids[0]=$!
	#run_sysfs_gim_info &
	#pids[1]=$!
	#gim_load_and_reload & 
	#pids[2]=$!
	#stop_all_test "${pids[*]}" 
	#do_cleanup

	echo load gim
	total_vm_to_run=$(( $vf_num_per_gpu * $total_gpu_num_support_sriov))
    echo "GG: total_vm_to_run: $total_vm_to_run"
	LOAD_GIM vf_num=$vf_num_per_gpu
	echo "Create VMs & assign VFs to VMs"
	create_vm_from_base $total_vm_to_run 2 $test_case_name
	echo "Waiting for the VM to up..."
	wait_mutli_vm_power_on $total_vm_to_run $test_case_name

	#run_sysfs_gim_info  &
    
    for i in $(seq 1 3)
    do
        run_get_vf
    
        for((vm_index=0; vm_index<$total_vm_to_run; vm_index++))
        do
            get_vm_name $vm_index
            #run_command_in_guest $vm_name "sudo sh -c 'cat /dev/null>/var/log/kern.log'"
            run_command_in_guest $vm_name "sudo modprobe amdgpu"
            run_command_in_guest $vm_name "sudo modprobe -r amdgpu"
        done
        echo "calling run_clear_vf_fb..."
        run_clear_vf_fb
        echo "calling run_hot_reset..."
        run_hot_reset
   done
    
	add_test_end_time_to_report
	echo test result: Pass | tee -a $test_report_folder/$report_host_file_name
  
}

do_cleanup
test_attack_sysfs

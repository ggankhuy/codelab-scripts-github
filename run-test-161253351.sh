BASIC_QUERY_PATH=/usr/src/gim-2.0.1.G.20201023/smi-lib/examples/basic-query
ROOT_DIR=`pwd`
counter=0
DATE=`date +%Y%m%d-%H-%M-%S`
mkdir $ROOT_DIR/log/

# 4 - guest driver reload 1
# 5 - guest driver reload 2
# 18 - reboot 1
# 38 - 3dmark
# 41 - yeti vk example
# 43 - xgemm
# 42 - yeti raw vk example
for i in 4 5 18 38 41 43 42
do

	echo "test $i counter $counter" 
	echo "GG:  test $i counter $counter" > /dev/kmsg
	sleep 3
	for j in $(seq 1 4)
		do virsh destroy vats-test-0$j
	done
	cd  $BASIC_QUERY_PATH
	make
	./alloc_vf_with_parameters > $ROOT_DIR/log/alloc_vf_parameters.loop-$counter.test-$i.log
	cd $ROOT_DIR
	./run-test.sh $i > $ROOT_DIR/log/vats2.loop-$counter.test-$i.log
    dmesg > $ROOT_DIR/log/vats2.loop-$counter.test-$i.dmesg.log
	counter=$((counter+1))
done

: <<'END'


1: Host Driver Reload 1
2: Host Driver Reload 2
3: Host Driver Capability
4: Guest Driver Reload 1
5: Guest Driver Reload 2
6: Guest Driver Reload 3
7: Guest Driver Compatibility
8: Power Consumption
9: DPM
10: One VF Mode
11: Performance
12: Performance Delta VM
13: Performance Delta GPU
14: FB Scan
15: VM Shutdown 1
16: VM Shutdown 2
17: VM Shutdown 3
18: VM Reboot 1
19: VM Reboot 2
20: VM Mixed Reboot
21: VM Destroy 1
22: VM Destroy 2
23: VM Destroy 3
24: VM Destroy 4
25: VM Pause 1
26: VM Pause 2
27: VM Pause 3
28: VM Pause 4
29: VM Reset 1
30: VM Reset 2
31: VM Reset 3
32: VM Reset 4
33: Game Session
34: Disaster Memory Leak
35: Benchmark Kill
36: Benchmark RGP
37: Benchmark GPA
38: Benchmark Yeti3DMark
39: Benchmark VkCts
40: Benchmark VkCtsMini
41: Benchmark YetiVkExample
42: Benchmark RawVkExample
43: Benchmark Xgemm
44: Benchmark Clinfo
45: Benchmark DGMA
46: Benchmark Vulkaninfo
47: Benchmark VideoEncodeTest
48: Benchmark VideoEncodeTestMini
49: Benchmark VulkanCTS
50: Vf Number Option
51: VM Reboot 3
52: TDR 1
53: TDR 1 SMI
54: TDR Dev 1
55: TDR Stress
56: TDR 2
57: TDR Dev 2
58: TDR 3 SMI
59: Mailbox Attack 1
60: Mailbox Attack 3
61: TESTSMC Error
62: TDR + Benchmark
63: Event Guard 5
64: Event Guard SMI 1
65: Event Guard SMI 2
66: Event Guard SMI 3
67: Event Guard SMI 4
68: MMIO Attack
69: MARC Write
70: MARC Write-Offset
71: MARC Write-Forbidden
72: MARC Write-QXL
73: MARC Read
74: TDR All Coverage
75: Guest DPM Attack
76: PCIe Configuration
77: GPU Monitor 1
78: GPU Monitor 2
79: GPU Monitor 3
80: GPU Monitor SMI 1
81: GPU Monitor SMI 3
82: Multiprocess Benchmark
83: Dump Tool
84: Host SMI Integration
85: MultiVfAssign
86: VBIOS Hash
87: Dynamic Clock Power
88: MMSCH VF Gate
89: PSP VF Gate
90: Flexible Benchmark
91: GIML Memory Leak
92: Attack SMI 1
93: Attack SMI 2
94: Attack SMI 3

END

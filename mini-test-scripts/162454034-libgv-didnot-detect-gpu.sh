#	Thipt only works for semitrucks where bmc is accessible through ssh.

# GPU flash range
CONFIG_GPU_FLASH_IDX_MIN=2
CONFIG_GPU_FLASH_IDX_MAX=13

# amdvbflash path
CONFIG_PATH_AMDVBFLASH=/root/tools/amdvbflash/amdvbflash-4.74

#  BMC access

CONFIG_BMC_IP="10.216.52.241"
CONFIG_BMC_USERNAME=root
CONFIG_BMC_PW=OpenBmc

# host ip access

CONFIG_OS_IP="10.216.52.232"
CONFIG_OS_USERNAME=root
COFNIG_OS_PW=amd1234

# POWER CYCLE TYPE

CONFIG_PC_REBOOT=1

#	power off and on. Interval between off and on is dictated by CONFIG_PC_POWERCYCLE_IN in seconds.

CONFIG_PC_POWERCYCLE=2 
CONFIG_PC_POWERCYCLE_INTERNAL=1

#	number of test to repeat

CONFIG_ITER=100 









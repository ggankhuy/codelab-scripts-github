# trigger surprise link down using pcie config register:
# 1 find pcie express capability register.
# 2 offset 10h/word size rx in pcie link capability.
# 3 set bit 4.

# warning: this wil make the device fall of pcie tree and make sure to use 
# it carefully.

# usage:
# ./surprise-link-down.sh bug dev fcn  (decimal)
# example:
# ./surprise-link-down.sh 0 08 03 (cause SLD in bus 0, device 08, function 03)
# ./surprise-link-down.sh 0 12 03 (cause SLD in bus 0, device 12 (0xc), function 03)


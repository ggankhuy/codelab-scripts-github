set -x
var3=`echo "ibase=16; FF" | bc`
var4=`echo $((var3 & 0xf0))`



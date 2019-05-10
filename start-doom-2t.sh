# 	fixed everything.
#	problem is at the end of each of t1 and t2 execution, different present directory is expected so 
#	this script may be just toast. 
/bin/bash ./yeti-game-test.sh doom yeti 2 t1 > t1.log && cat t1.log | tail -n 2 &
/bin/bash ./yeti-game-test.sh doom yeti 2 t2 > t2.log && cat t2.log | tail -n 2 &

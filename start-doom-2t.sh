/bin/bash ./yeti-game-test.sh doom yeti 2 t1 > t1.log 
/bin/bash ./yeti-game-test.sh doom yeti 2 t2 > t2.log 
echo =======================
cat t1.log | grep tail -n 2
echo =======================
cat t2.log | grep tail -n 2
echo =======================

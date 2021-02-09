cd /git.co/ad-hoc-scripts/mini-test-scripts/161253351
i=0; 
while true; 
do echo "monitor: $i" > monitor.log ; ./monitor > monitor.log ;i=$((i+1)); 
done


i=0; 
while true; 
do echo "monitor: $i" > monitor.log ; ./monitor > monitor.log ;i=$((i+1)); 
done


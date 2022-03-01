if [[ -z `which csvtool` ]] ; then apt install csvtool -y ; fi
FILE1=p61-rocprof
FILE2=p61-multi
for i in $FILE1 $FILE2 ; do
    hipcc $i.cpp
done
OUTPUT1=rocprof-test.csv
OUTPUT2=rocprof-test2.csv

if [[ $? -eq 0 ]] ; then
    rocprof -i input.xml -o $OUTPUT1 ./a.out
    csvtool readable $OUTPUT1
else 
    echo "Compile failed."
fi

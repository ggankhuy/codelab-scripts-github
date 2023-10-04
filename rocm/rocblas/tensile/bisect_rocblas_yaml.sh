FILENAME=arcturus_Cijk_Ailk_Bjlk_DB
FILENAME=aldebaran_Cijk_Ailk_Bjlk_DB

lineNo=0
solIndexNo=0
OUT_DIR=output
mkdir -p $OUT_DIR
while IFS= read -r line

do
    outputFile=$FILENAME.$solIndexNo.out.yaml
    if [[ `echo $line | grep SolutionIndex` ]] ; then
        echo "$line"
        solIndexNo=$((solIndexNo+1))
    fi
    lineNo=$((lineNo+1))
    echo $line >> $outputFile
done < "$FILENAME.yaml"

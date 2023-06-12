wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
chmod a+x /usr/local/bin/yq

if [[ -z `which yq` ]] ; then
    echo "Unable to install yq"
    exit 1
fi

input=arcturus_Cijk_Ailk_Bjlk_SB.yaml
interim=arcturus_Cijk_Ailk_Bjlk_SB.0.yaml
output=arcturus_Cijk_Ailk_Bjlk_SB_filtered.yaml

# go until line reading [.*[0-9].*,.*[0-9].*\]" but not 4 repeated.
# if first number is less than 10, then record current and prev. line.

prev_line=""
echo -ne "" > $output

idx=0
solIdx=1

for idx in {0..4}; do
    echo -ne "  - " >> $output ;
    yq eval -M '.['$idx']' $input >> $output
done

echo -ne "  - " >> $output
idx=0
while [[ $solIdx ]] ; do
    echo "processing [5][$idx]..."
    solIdx=`yq eval -M '.[5]['$idx']' $input | grep SolutionIndex | tr -s ' ' | cut -d " " -f2`
    echo "[5]: SolutionIndex: $solIdx"
    if [[ $solIdx == "null" ]] ; then
        break
    fi

    if [[ $solIdx -le 10 ]] ; then
        echo "outputting to interm..."
        echo -ne "  - " >> $output ;
        yq eval -M '.[5]['$idx']' $input >> $output
    fi
    idx=$((idx+1))
done

yq eval -M '.[6]' $input >> $output

while [[ $solIdx ]] ;  
do
    echo "..."
    echo "$idx"
    solIdx=`yq eval -M '.[7]['$idx'][1][0]' $input`
    echo "solIdx: $solIdx" 
    if [[ $solIdx == "null" ]] ; then
        break
    fi
    yq eval -M '.[7]['$idx']' $input

    if [[ $solIdx -le 10 ]] ; then
        echo "outputting to interm..."
        echo -ne "  - - " >> $output ; 
        yq eval -M '.[7]['$idx'][0]' $input | tee -a $output
        echo -ne "    - " >> $output ; 
        yq eval -M '.[7]['$idx'][1]' $input | tee -a $output
    fi
    idx=$((idx+1))
done
yq $output -o $output.json

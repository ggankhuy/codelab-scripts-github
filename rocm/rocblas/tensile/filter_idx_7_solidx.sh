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
echo -ne "" > $interim

idx=0
solIdx=1

while [[ $solIdx ]] 
do
    echo "..."
    solIdx=`yq eval -M '.[7]['$idx'][1][0]' $input`
    echo "solIdx: $solIdx"
    idx=$((idx+1))
    echo "processing"
    yq eval -M '.[7]['$idx']' $input
    sleep 1
done

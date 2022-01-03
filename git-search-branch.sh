for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "file="` ]]  ; then
        echo "filename: $var"
        FILE_NAME=`echo $var | cut -d '=' -f2`
    fi
done

BRANCH_LIST_LOG=blist.log
RESULT_LOG=git-search-result.log
sudo git branch -r | sudo tee -a $BRANCH_LIST_LOG

IFS=$'\n' read -d '' -r -a branch_list < $BRANCH_LIST_LOG
echo "Reading branch list from log now..."
counter=0
echo "" |  sudo tee $RESULT_LOG
for i in ${branch_list[@]}
do 
    i=`echo $i | cut -d '/' -f2`
    echo branch: $i
    counter=$((counter+1))

    echo "------------------" | sudo tee -a $RESULT_LOG
    echo "Switching to branch : $i..." | sudo tee -a $RESULT_LOG
    sudo git checkout $i
    find $FILE_NAME
    if [[ $? -ne 0 ]] ; then 
        echo "Could not find file" | sudo tee -a $RESULT_LOG 
    else
        echo "Found the file in branch $i..." | sudo tee -a  $RESULT_LOG
        break
    fi
    
    # this line is for test only for limited iteration.
    #if [[ $counter -gt 5 ]]  ; then break ; fi
done

echo "Done searching, please look into $RESULT_LOG for search result."

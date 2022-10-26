for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "file="` ]]  ; then
        FILE_NAME=`echo $var | cut -d '=' -f2`
    fi
    if [[ ! -z `echo "$var" | grep "commit="` ]]  ; then
        COMMIT_ID=`echo $var | cut -d '=' -f2`
    fi
done

echo "FILENAME: $FILENAME"
echo "COMMIT_ID: $COMMIT_ID"

BRANCH_LIST_LOG=blist.log
RESULT_LOG=git-search-result.log
sudo git branch -r | sudo tee -a $BRANCH_LIST_LOG

IFS=$'\n' read -d '' -r -a branch_list < $BRANCH_LIST_LOG
echo "Reading branch list from log now..."
counter=0
echo "" |  sudo tee $RESULT_LOG

counter=0
for i in ${branch_list[@]}
do 
    i=`echo $i | cut -d '/' -f2`
    echo branch: $i
    counter=$((counter+1))

    echo "$counter. ------------------" | sudo tee -a $RESULT_LOG
    echo "Switching to branch : $i..." | sudo tee -a $RESULT_LOG
    sudo git checkout $i
    if [[ ! -z $FILENAME ]] ; then
        find $FILE_NAME
        if [[ $? -ne 0 ]] ; then 
            echo "Could not find file" | sudo tee -a $RESULT_LOG 
        else
            echo "Found the file in branch $i..." | sudo tee -a  $RESULT_LOG
            break
        fi
    
        # this line is for test only for limited iteration.
        #if [[ $counter -gt 5 ]]  ; then break ; fi
    fi
    if [[ ! -z $COMMIT_ID ]] ; then
        git log | grep $COMMIT_ID
        if [[ $? -ne 0 ]] ; then 
            echo "Could not find commit id" | sudo tee -a $RESULT_LOG 
        else
            echo "Found the commit ID in branch $i..." | sudo tee -a  $RESULT_LOG
        fi
    fi
done

echo "Done searching, please look into $RESULT_LOG for search result."

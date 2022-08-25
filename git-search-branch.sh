for var in "$@"
do
    find $FILE_NAME
    if [[ ! -z `echo "$var" | grep "url="` ]]  ; then
        echo "url: $var"
        URL=`echo $var | cut -d '=' -f2`
    fi
    if [[ ! -z `echo "$var" | grep "file="` ]]  ; then
        echo "file: $var"
        FILE_NAME=`echo $var | cut -d '=' -f2`
    fi
    if [[ ! -z `echo "$var" | grep "commit="` ]]  ; then
        echo "commit: $var"
        COMMIT_ID=`echo $var | cut -d '=' -f2`
    fi
done
LOG_FOLDER=`pwd`/log/
echo "LOG_FOLDER: $LOG_FOLDER"
sleep 2
mkdir $LOG_FOLDER -p
BRANCH_LIST_LOG=$LOG_FOLDER/blist.log
RESULT_LOG=$LOG_FOLDER/search-result.log
echo -ne "" > $RESULT_LOG
echo -ne "" > $BRANCH_LIST_LOG
#git clone $URL
echo "url: $URL"
FOLDER=`echo $URL  | grep -o '[^/]*$' | cut -d '.' -f1`
echo "folder: $FOLDER"
cd $FOLDER

if [[ $? -ne 0 ]] ; then
    echo "Unable to cd into $FOLDER"
    exit 1
fi
pwd
sleep 2
sudo git branch -r | sudo tee $BRANCH_LIST_LOG
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
    if [[ $FILE_NAME ]] ; then
        echo "Looking for file: $FILENAME..."
        find $FILE_NAME
        if [[ $? -ne 0 ]] ; then 
            echo "Could not find file" | sudo tee -a $RESULT_LOG 
        else
            echo "Found the file in branch $i..." | sudo tee -a  $RESULT_LOG
            break
        fi
    fi

    if [[ $COMMIT_ID ]] ; then
        echo "Looking for commit $COMMIT_ID..."
        git --no-pager log --oneline | grep $COMMIT_ID
        if [[ $? -ne 0 ]] ; then 
            echo "Could not find the commit " | sudo tee -a $RESULT_LOG
        else
            echo "Found the commit in branch $i..." | sudo tee -a $RESULT_LOG
        fi
    fi
    
    # this line is for test only for limited iteration.
    #if [[ $counter -gt 5 ]]  ; then break ; fi
done

echo "Done searching, please look into $RESULT_LOG for search result."

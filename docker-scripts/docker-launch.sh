# check if ML container exists.
# if not create one
    # check if ML image exists.
    # create one and run init.sh scripts from volume.
#launch container and attach terminal.


function usage (){
    echo "Usage: $0 --ml <tensorflow|tf|pytorch|py>"
}

for var in "$@"
do
    if [[ ! -z `echo "$var" | grep "ml="` ]]  ; then
        echo "ML selected: $var"
        ML=`echo $var | cut -d '=' -f2`
    fi
done

case "$ML" in
   tensorflow|tf)
        echo "Selected tf..."
        ML=tensorflow
      ;;
   pytorch|py)
        echo "Selected py..."
        ML=pytorch
      ;;
   "")
        echo "ml is not specified, defaulting to py..."
        ML=pytorch
      ;;
   *)
     usage; exit 1
     ;;
esac

echo "ML: $ML"
VOLUME_NAME=volume-rocm-$ML-persistent
sudo docker volume ls | grep $VOLUME_NAME

ret=$?
echo "ret: $ret"
if [[ $ret==0  ]] ; then
    echo "volume $VOLUME_NAME already exists. Launching container with this volume..."
else
    echo "volume $VOLUME_NAME does not exist. Creating one..."    
    sudo docker volume create --name $VOLUME_NAME
fi

containerID=`sudo docker container ps -a | grep $ML:latest | cut -d ' ' -f1`
echo "containerID: $containerID"

if [[ -z $containerID ]] ; then
    echo "Container does not exist."
    sudo docker run -d -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx --mount source=$VOLUME_NAME,target=/data \
    rocm/$ML:latest

    # Attempt container ID again.

    containerID=`sudo docker container ps -a | grep $ML:latest | cut -d ' ' -f1`

    if [[ -z $containerID ]] ; then echo "Failed to get container ID..." ; exit 1 ; fi

    # Copy init script and run it (apt install pytorch install etc...)
fi
echo "Starting container $containerID..."
sudo docker container start $containerID
echo "Attaching to container $containerID" 
sudo docker attach $containerID



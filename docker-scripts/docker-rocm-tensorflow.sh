VOLUME_NAME=rocm-tensorflow-persistent
sudo docker volume ls | grep $VOLUME_NAME

ret=$?
echo "ret: $ret"
if [[ $ret==0  ]] ; then
    echo "volume $VOLUME_NAME already exists. Launching container with this volume..."
else
    echo "volume $VOLUME_NAME does not exist. Creating one..."    
    sudo docker volume create --name $VOLUME_NAME
fi
alias drun='sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx --mount source=$VOLUME_NAME,target=/data'
drun rocm/tensorflow:latest


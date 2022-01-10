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
VOLUME_NAME=rocm-$ML-persistent
sudo docker volume ls | grep $VOLUME_NAME

ret=$?
echo "ret: $ret"
if [[ $ret==0  ]] ; then
    echo "volume $VOLUME_NAME already exists. Launching container with this volume..."
else
    echo "volume $VOLUME_NAME does not exist. Creating one..."    
    sudo docker volume create --name $VOLUME_NAME
fi
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx --mount source=$VOLUME_NAME,target=/data \
rocm/$ML:latest


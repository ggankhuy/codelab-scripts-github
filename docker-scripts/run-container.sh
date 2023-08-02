set +x
if [[ -z $1 ]] ; then
    echo "Usage: "
    echo "To create container: $0 <docker_image_name> <container_name>"
    echo "To start container: $0 <container_name>"
    exit 1
fi

if [[ $2 ]] ; then
    sudo docker run -it \
        --network=host \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add=video \
        --ipc=host \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --name=$2 -v $HOME/dockerx:/dockerx $1 
    if [[ $? -ne 0 ]];  then
        echo "It appears container already exists, starting and logging to container..."
        sudo docker start $2 ; sudo docker exec -it $2 bash
    fi
else
    sudo docker start $1 ; sudo docker exec -it $1 bash
fi



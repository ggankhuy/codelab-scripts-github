set +x
if [[ -z $1 ]] || [[ -z $2 ]] ; then
    echo "Usage: $0 <docker_image_name> <conainer_name>"
    exit 1
fi
sudo docker run -it --network=host --device=/dev/kfd --device=/dev/dri --group-add=video --ipc=host --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --name=$2 -v $HOME/dockerx:/dockerx $1 


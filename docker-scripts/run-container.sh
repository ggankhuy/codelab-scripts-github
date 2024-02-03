set -x

DEFAULT_GPU="amd"

if [[ -z $1 ]] ; then
    echo "Usage: "
    echo "To create container: $0 --name=<container_name> --image=<container_image> --gpu=<rocm,cuda,amd,nvidia>"
    echo "To start container: $0 --name=<container_name>"   
    exit 0
fi

for var in "$@"
do
    echo var: $var

    if [[ $var == *"--name="* ]]  ; then
        name=`echo $var | cut -d '=' -f2`
    fi

    if [[ $var == *"--image="* ]]  ; then
        image=`echo "$var" | cut -d '=' -f2`
    fi

    if [[ $var == *"--gpu="* ]]  ; then
        gpu=`echo $var | cut -d '=' -f2`
    fi
done 

echo "name: $name"
echo "image: $image"
echo "gpu: $gpu"

#   Must be specified: container name.

if [[ -z $name ]] ; then
    echo "Error: container name must be specified."
    exit 1
fi

#   Attempt to start container.

#ret=`sudo docker container list --all | grep $name | cut -d '=' -f5`

if [[ ! -z $image ]] ; then
    echo "Image is specified, therefore will attempt to create container..."

    # create based on amd or nvidia:

    if [[ -z $gpu ]] ; then
        echo "Gpu has not been specified, will default to $DEFAULT_GPU..."
        gpu=$DEFAULT_GPU
    fi

    if [[ $gpu == "amd" ]] || [[ $gpu == "rocm" ]] ; then
        sudo docker run -it \
            --network=host \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add=video \
            --ipc=host \
            --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            -v vol-$name:/root/extdir \
            -w /root/ \
            --privileged \
            --name=$name $image 
    elif [[ $gpu == "nvidia" ]] || [[ $gpu == "cuda" ]] ; then
        echo "nvidia selected, docker run command constructed: "
        sudo docker run -it --runtime=nvidia --gpus all --name=$name ubuntu
    else
        echo "Error: Unknown gpu specified: $gpu"
        exit 1
    fi

    if [[ $? -ne 0 ]] ; then
        echo "Create docker failed, it may already exist, will attempt starting..."
        sudo docker start $name ; sudo docker exec -it $name bash
        ret=$?
        if [[ $ret -ne 0 ]] ; then
            echo "Start docker failed, something wrong! return code: $ret"
            exit 1
        fi
    fi
else
    echo "Image not specified, will attempt starting container..."
    sudo docker start $name ; sudo docker exec -it $name bash
    ret=$? 
    if [[ $ret -ne 0 ]] ; then
        echo "Start docker failed, something wrong! return code: $ret"
        exit 1
    fi
fi



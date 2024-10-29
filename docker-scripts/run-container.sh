<<<<<<< HEAD
set -x
=======
set +x
>>>>>>> 0d653266ab71cfdb5b0a9c1b5d3e0e6224399362
if [[ $? -ne 0 ]] ; then echo "either driver failed to load or docker service failed to start. Check logs" ; exit 1 ;  fi

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

OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`

if [[ ! -z $image ]] ; then
    echo "Image is specified, therefore will attempt to create container..."

    # create based on amd or nvidia:

    if [[ -z $gpu ]] ; then
        echo "Gpu has not been specified, will default to $DEFAULT_GPU..."
        gpu=$DEFAULT_GPU
        sudo modprobe amdgpu && sudo systemctl start docker

    fi

    if [[ $gpu == "amd" ]] || [[ $gpu == "rocm" ]] ; then
        sudo modprobe amdgpu && sudo systemctl start docker
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
        sudo systemctl start docker
        echo "nvidia selected, docker run command constructed: "

        # setup nvidia docker support 

        echo "OS_NAME: $OS_NAME"
        case "$OS_NAME" in
            "Ubuntu")
                echo "Ubuntu is detected..."
                PKG_EXEC=apt
                curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
                    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
                curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
                sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
                sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
                sudo apt-get update
                sudo apt-get install -y nvidia-container-toolkit
                ;;
           "CentOS Linux")
                echo "CentOS is detected..."
                PKG_EXEC=yum
                curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
                    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
                sudo yum-config-manager --enable nvidia-container-toolkit-experimental
                sudo yum install -y nvidia-container-toolkit
                sudo nvidia-ctk runtime configure --runtime=docker
                sudo systemctl restart docker
                ;;
           "CentOS Stream")
                echo "CentOS is detected..."
                PKG_EXEC=yum
                ;;
           *)
             echo "Unsupported O/S, exiting..." ; exit 1
             ;;
        esac


        sudo docker run -it \
            --network=host \
            -v vol-$name:/root/extdir \
            -w /root/ \
            --privileged \
            --runtime=nvidia \
            --gpus all \
            --name=$name $image
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
    if [[ $gpu == "amd" ]] || [[ $gpu == "rocm" ]] ; then
        sudo modprobe amdgpu
    fi
    sudo systemctl start docker && sudo docker start $name && sudo docker exec -it $name bash
    ret=$? 
    if [[ $ret -ne 0 ]] ; then
        echo "Start docker failed, something wrong! return code: $ret"
        exit 1
    fi
fi



yum update -y ; yum install cmake git tree nano wget g++ python3-pip 
sudo dnf install epel-release epel-next-release -y ; sudo dnf config-manager --set-enabled crb ; sudo dnf install epel-release epel-next-release -y
cd ~/exdir ; mkdir gg; cd gg ; mkdir git log wget back transit ; cd git ; echo "cd `pwd`" >> ~/.bashrc

if [[ -z $1 ]] ; then
    echo "Usage: "
    echo "$0 --ga --version=6.0 / use ga version of rocm version 6.0"
    echo "$0 --int --mainline --build=13435 --amdgpu=1720120 / use internal version of rocm"
    echo "$0 --int --release --build=91 --version --amdgpu=1720120 / use internal version of rocm"

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

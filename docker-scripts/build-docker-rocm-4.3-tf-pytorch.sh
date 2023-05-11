DOCKER_USER=ggdocker
DOCKER_PASS=8981555
ROCKER_REPO_NAME=docker-rocm
TAG=u1804-rocm-4.3
docker pull ubuntu:18.04
docker run -d --name ubuntu-1804 -h u1804 -e LANG=C.UTF-8 -it ubuntu:18.04
CONTAINER_ID=`docker container ls | grep 18.04 | tr -s ' ' | cut -d ' ' -f1`
echo "container id: $CONTAINER_ID"
docker exec -it $CONTAINER_ID bash -c "apt-get update"
docker exec -it $CONTAINER_ID bash -c "apt install git tree net-tools apt-utils gnupg wget python3-pip -y"
docker exec -it $CONTAINER_ID bash -c "apt remove amdgpu-dkms -y"
docker exec -it $CONTAINER_ID bash -c "apt remove amdgpu-dkms-firmware -y"
docker exec -it $CONTAINER_ID bash -c "apt update -y ; apt dist-upgrade  -y"
docker exec -it $CONTAINER_ID bash -c "apt install libnuma-dev -y"
docker exec -it $CONTAINER_ID bash -c "wget -qO - http://repo.radeon.com/rocm/apt/4.3/rocm.gpg.key | apt-key add -"
docker exec -it $CONTAINER_ID bash -c "echo 'cd ~/ROCm/' >> ~/.bashrc"
docker exec -it $CONTAINER_ID bash -c "echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/4.3/ xenial main' | tee /etc/apt/sources.list.d/rocm.list"
docker exec -it $CONTAINER_ID bash -c "apt update ; apt install rocm-dkms -y"
docker exec -it $CONTAINER_ID bash -c "pip3 install pip --upgrade"
docker exec -it $CONTAINER_ID bash -c "pip3 install tensorflow-rocm numpy pandas matplotlib sklearn ; pip3 list "
docker login --username=$DOCKER_USER --password=$DOCKER_PASS
docker commit debian-buster-slim $DOCKER_USER/$DOCKER_REPO_NAME[:$TAG]



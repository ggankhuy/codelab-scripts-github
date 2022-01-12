DOCKER_USER=ggdocker
DOCKER_PASS=8981555

function docker_install_apt() {
    sudo apt-get remove docker docker-engine docker.io containerd runc
    sudo apt-get update
    sudo apt-get install \
    ca-certificates \
    curl \
    gnupg \
    lsb-release -y
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install docker-ce docker-ce-cli containerd.io -y
}

function docker_install_yum() {
    sudo yum install epel-release -y
    #sudo yum-config-manager --enable centos-extras
    sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-engine
    sudo yum install -y yum-utils -y
    sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
    sudo yum-config-manager --enable docker-ce-test
    sudo yum install docker-ce docker-ce-cli containerd.io --allowerasing -y
}

OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
echo OS_NAME: $OS_NAME
case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      docker_install_apt
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      docker_install_yum
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

sudo systemctl start docker
sudo docker login --username=$DOCKER_USER --password=$DOCKER_PASS
sudo docker run hello-world

FROM python:3.6
RUN \
#   echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic main restricted" >> /etc/apt/sources.list && \
#   echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic-updates main restricted" >> /etc/apt/sources.list && \
#   echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic universe" >> /etc/apt/sources.list && \
#   echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic-updates universe" >> /etc/apt/sources.list && \
#   echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic multiverse" >> /etc/apt/sources.list && \
#   echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic-updates multiverse" >> /etc/apt/sources.list && \
#   echo "deb http://us.archive.ubuntu.com/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
#   echo "deb http://security.ubuntu.com/ubuntu bionic-security main restricted" >> /etc/apt/sources.list && \
#   echo "deb http://security.ubuntu.com/ubuntu bionic-security universe" >> /etc/apt/sources.list && \
#   echo "deb http://security.ubuntu.com/ubuntu bionic-security multiverse" >> /etc/apt/sources.list && \
    cat /etc/apt/sources.list && apt-get update && \
#    apt install -y git tree net-tools && \
    apt remove amdgpu-dkms -y && \
    apt remove amdgpu-dkms-firmware -y && \
    apt update -y && \ apt dist-upgrade -y && \
    apt install libnuma-dev -y  && \
    wget -qO - http://repo.radeon.com/rocm/apt/debian/rocm.gpg.key | sudo apt-key add - && \
    echo 'cd ~/ROCm/' >> ~/.bashrc && \
    echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/debian/ xenial main' | sudo tee /etc/apt/sources.list.d/rocm.list && \
    apt update && apt install rocm-dkms -y


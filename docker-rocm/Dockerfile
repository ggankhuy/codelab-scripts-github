FROM python:3.6
RUN apt-get update 
RUN apt install -y git tree net-tools apt-utils
RUN apt remove amdgpu-dkms -y ; echo optional
RUN apt remove amdgpu-dkms-firmware -y ; echo optional
RUN apt update -y ; apt dist-upgrade -y
RUN apt install libnuma-dev -y 
RUN wget -qO - http://repo.radeon.com/rocm/apt/4.3/rocm.gpg.key | apt-key add - 
RUN echo 'cd ~/ROCm/' >> ~/.bashrc 
RUN echo 'deb [arch=amd64] http://repo.radeon.com/rocm/apt/4.3/ xenial main' | tee /etc/apt/sources.list.d/rocm.list 
RUN apt update && apt install rocm-dkms -y


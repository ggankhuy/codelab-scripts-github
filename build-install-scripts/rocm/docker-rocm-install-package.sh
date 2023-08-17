# package based installation.

set -x
export SUDO=sudo;
echo SUDO: $SUDO
$SUDO apt-get update -y &&\
$SUDO apt install sudo tree nano git wget python3-pip curl gpg -y && \
$SUDO mkdir --parents --mode=0755 /etc/apt/keyrings && \
$SUDO wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
$SUDO gpg --dearmor | $SUDO tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/5.6/ubuntu focal main' | \
$SUDO tee /etc/apt/sources.list.d/amdgpu.list && $SUDO apt update -y && \
for ver in 5.3.3 5.4.3 5.5.1 5.6 ; do echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ver focal main" | \
$SUDO tee --append /etc/apt/sources.list.d/rocm.list ;  done && \
echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | $SUDO tee /etc/apt/preferences.d/rocm-pin-600 && \
$SUDO apt update -y && \
$SUDO apt install rocm-hip-sdk -y && \
$SUDO  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6 && \
python3 -c "import torch ; print(torch.cuda.is_available())"

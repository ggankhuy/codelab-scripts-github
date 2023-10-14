set -x

# This script is based on rocm5.6. If other versions, need modifications!

# package based installation.
INSTALL_MODE_PACKAGE=1
INSTALL_MODE_SCRIPT=2
INSTALL_MODE=$INSTALL_MODE_SCRIPT

export SUDO=sudo;
echo SUDO: $SUDO
$SUDO apt-get update -y &&\
$SUDO apt install sudo tree nano git wget python3-pip curl gpg -y && \

case "$INSTALL_MODE" in
    $INSTALL_MODE_SCRIPT )
        echo "Using script based installation..."
        sudo wget https://repo.radeon.com/amdgpu-install/5.6/ubuntu/jammy/amdgpu-install_5.6.50600-1_all.deb
        apt install -y ./amdgpu-install_5.6.50600-1_all.deb
        amdgpu-install --usecase=rocm --no-dkms -y
        ;;

    $INSTALL_MODE_PACKAGE )
        echo "Using package based installation..."
        $SUDO mkdir --parents --mode=0755 /etc/apt/keyrings && \
        $SUDO wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
        $SUDO gpg --dearmor | $SUDO tee /etc/apt/keyrings/rocm.gpg > /dev/null && \
        echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/amdgpu/5.6/ubuntu focal main' | \
        $SUDO tee /etc/apt/sources.list.d/amdgpu.list && $SUDO apt update -y && \
        for ver in 5.3.3 5.4.3 5.5.1 5.6 ; do echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ver focal main" | \
        $SUDO tee --append /etc/apt/sources.list.d/rocm.list ;  done && \
        echo -e 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | $SUDO tee /etc/apt/preferences.d/rocm-pin-600 && \
        $SUDO apt update -y && \
        $SUDO apt install rocm-hip-sdk -y
        ;;

    *)
        echo "Unknown installation method is set: $INSTALL_MODE. Exiting..."
        exit 1
    ;;
esac

if [[ $CONFIG_INSTALL_PYTORCH ]] ; then
    echo "Installing pytorch."
    $SUDO  pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm5.6 && \
    python3 -c "import torch ; print(torch.cuda.is_available())"
fi




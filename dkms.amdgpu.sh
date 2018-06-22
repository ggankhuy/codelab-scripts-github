VERSION=amdgpu/5.16.9.22.20-1423825.el8
KERNEL_VERSION=5.12.0-0_fbk5_zion_rc1_5697_g2c723fb88626
dkms remove  $VERSION -k $(uname -r)/x86_64
dkms build $VERSION -k $(uname -r)/x86_64 --kernelsourcedir=/usr/src/kernels/$KERNEL_VERSION
dkms install  $VERSION -k $(uname -r)/x86_64

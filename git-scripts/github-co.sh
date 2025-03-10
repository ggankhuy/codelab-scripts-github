CONFIG_USE_SSH_AMD_CLOUD_GPU=0
#git config credential.helper store

if [[ -z `cat ~/.git-credentials | grep ggghamd`  ]] ; then
    echo "storing credentials for ggghamd@github.com"
    echo "https://ggghamd:amd1234A%23@github.com" >> ~/.git-credentials
else
    echo "credentials for ggghamd@github.com is already stored, continuing..."
fi

if [[ $CONFIG_USE_SSH_AMD_CLOUD_GPU -eq 1 ]] ; then
    amd_cloud_gpu_repos=(\
    git@github.com:AMD-CloudGPU/Gibraltar-LibGV.git
    )
else
    amd_cloud_gpu_repos=(\
#    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-LinuxGuestKernel.git \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-Vulkan \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-GIM \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-Libdrm \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-Libdrm-Public \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-LibGV.git \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-misc.git \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/Gibraltar-VkExamples \
    https://ggghamd:amd1234A%23@github.com/AMD-CloudGPU/SMI-Lib.git \
	
    )
fi

mkdir amd-gib
cd amd-gib

for i in ${amd_cloud_gpu_repos[@]}
do
    echo $i
    git clone $i
    sleep 1
done

cd  ..

amd_gerrit_repos=(\
ssh://gerritgit/gpu-virtual/ec/driver/libgv \
ssh://gerritgit/gpu-virtual/ec/tool/smi-lib \
ssh://gerritgit/gpu-virtual/ec/driver/vats2 \
ssh://gerritgit/gpu-virtual/ec/driver/vats \	
ssh://gerritgit/brahma/ec/drm \
ssh://gerritgit/brahma/ec/linux \
ssh://gerritgit/brahma/ec/umr \
ssh://gerritgit/gpu-virtual/ec/driver/vats2 \
ssh://gerritgit/gpu-virtual/ec/driver/gim \
ssh://gerritgit/android/ec/amd/vendor/amd/proprietary/bin/tools/quark \
)

mkdir gerritt
cd gerritt

for i in ${amd_gerrit_repos[@]}
do
    echo $i
    git clone $i
    sleep 1
done

cd ..

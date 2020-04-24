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
#    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-LinuxGuestKernel.git \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-Vulkan \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-GIM \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-Libdrm \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-LibGV.git \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-misc.git \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-VkExamples \
    )
fi

mkdir amd-gib
cd amd-gib

for i in ${amd_cloud_gpu_repos[@]}
do
    echo $i
    echo cloning $i...
    #git clone $i 
    ln ../expect.sh .
    export EXPECT_URL=$i
    echo "EXPECT_URL1: $EXPECT_URL"
    expect ./expect.sh
    #expect "Password for '$i':"
    #send -- "amd1234A#\r"
    #expect eof
done

cd  ..

amd_gerrit_repos=(\
ssh://gerritgit/gpu-virtual/ec/driver/libgv \
ssh://gerritgit/gpu-virtual/ec/tool/smi-lib \
ssh://gerritgit/gpu-virtual/ec/driver/vats2 \
ssh://gerritgit/gpu-virtual/ec/driver/vats \	
ssh://gerritgit/brahma/ec/drm \
ssh://gerritgit/gpu-virtual/ec/driver/gim \
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

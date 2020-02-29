CONFIG_USE_SSH_AMD_CLOUD_GPU=1

if [[ $CONFIG_USE_SSH_AMD_CLOUD_GPU -eq 1 ]] ; then
    amd_cloud_gpu_repos=(\
    git@github.com:AMD-CloudGPU/Gibraltar-LibGV.git
    )
else
    amd_cloud_gpu_repos=(\
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-LinuxGuestKernel.git \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-Vulkan \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-GIM \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-Libdrm \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-LibGV.git \
    https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-misc.git\
    )
fi

mkdir amd-gib
cd amd-gib

for i in ${amd_cloud_gpu_repos[@]}
do
    echo $i
    echo cloning $i...
    echo "amd1234A#" | git clone $i
    #expect "Password for 'https://ggghamd@github.com'"
    #expect "Password for"
    #send -- "amd1234A#\r" 
    sleep 1
done

cd  ..

amd_gerrit_repos=(\
ssh://gerritgit/gpu-virtual/ec/driver/libgv \
ssh://gerritgit/gpu-virtual/ec/tool/smi-lib \
ssh://gerritgit/gpu-virtual/ec/driver/vats2 \
ssh://gerritgit/gpu-virtual/ec/driver/vats \	
ssh://gerritgit/brahma/ec/drm \
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

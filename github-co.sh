amd_cloud_gpu_repos=(\
https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-LinuxGuestKernel.git \
https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-Vulkan \
https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-GIM \
https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-Libdrm \
https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-LibGV.git \
https://ggghamd@github.com/AMD-CloudGPU/Gibraltar-misc.git\
)

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
ssh://gerritgit/brahma/ec/drm \
ssh://gerritgit/brahma/ec/linux \
ssh://gerritgit/brahma/ec/umr \
ssh://gerritgit/gpu-virtual/ec/driver/vats2 \
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

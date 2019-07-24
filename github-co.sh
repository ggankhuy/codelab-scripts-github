amd_cloud_gpu_repos=(\
https://github.com/AMD-CloudGPU/Gibraltar-LinuxGuestKernel.git \
https://github.com/AMD-CloudGPU/Gibraltar-Vulkan \
https://github.com/AMD-CloudGPU/Gibraltar-GIM \
https://github.com/AMD-CloudGPU/Gibraltar-Libdrm \
https://github.com/AMD-CloudGPU/Gibraltar-LibGV.git \
https://github.com/AMD-CloudGPU/Gibraltar-misc.git\
)

mkdir amd-gib
cd amd-gib

for i in ${amd_cloud_gpu_repos[@]}
do
    echo $i
    git checkout $i
done



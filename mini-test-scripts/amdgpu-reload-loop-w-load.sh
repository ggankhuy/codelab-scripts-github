#   currently load is rochpl. can be replaed with other apps.
#   run from rochpl github root folder after building.
for i in {1..50}; do
    sudo dmesg --clear
    sudo modprobe amdgpu
    sudo dmesg 2>&1 | sudo tee -a log/dmesg.amdgpu.load.$i.log
    sudo OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 ./build/mpirun_rochpl -P 2 -Q 4 -N 430080 --NB 512 2>&1 | sudo tee log/mpirun_rochpl.$i.log
    sudo dmesg 2>&1 | sudo tee -a log/dmesg.rochpl.$i.log
    sudo dmesg --clear
    sudo modprobe -r amdgpu
    sudo dmesg 2>&1 | sudo tee -a log/dmesg.amdgpu.unload.$i.log
    sudo dmesg --clear
done




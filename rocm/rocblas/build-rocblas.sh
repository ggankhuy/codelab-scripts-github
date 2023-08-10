pushd /root/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/rocBLAS
t1=$((SECONDS)) ; ./install.sh -ida gfx908 -t ~/gg/git/codelab-scripts/build-install-scripts/rocm/ROCm-5.2/Tensile/ 2>&1 | tee build.log ; t2=$((SECONDS)) 
echo "time to build: $((t2-t1))
popd



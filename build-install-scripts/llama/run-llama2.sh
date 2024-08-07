set -x
cat /opt/rocm/.info/version
sudo dkms status
ls -l $LLAMA_PREREQ_PKGS.tar
tree -fs $LLAMA_PREREQ_PKGS
pip3 list | egrep "torch|triton|gradlib|vllm"
pushd $LLAMA_PREREQ_PKGS
./run_llama2_70b_bf16.sh && echo "---- done ----"   
popd


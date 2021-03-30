git config --global user.email "you@example.com"
git config --global user.name "Your Name"
mkdir -p ~/ROCm/
pushd  ~/ROCm/
mkdir -p ~/bin/
apt install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod a+x ~/bin/repo
~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-4.1.x
~/bin/repo sync
echo "ROCm source is downloaded to ~/ROCm"
echo "push ~/ROCm to get there..."
popd

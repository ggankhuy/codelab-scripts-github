git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git config --global color.ui false
mkdir -p ~/ROCm/
pushd  ~/ROCm/
mkdir -p ~/bin/
echo "install repo..."
apt install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod a+x ~/bin/repo
echo "repo init..."
~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-4.1.x
echo "repo sync..."
~/bin/repo sync
echo "ROCm source is downloaded to ~/ROCm"
echo "push ~/ROCm to get there..."
popd

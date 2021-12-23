p1=$1
CONFIG_VERSION=4.1
if [[ -z $p1 ]] ; then
    echo "Version not specified. Setting to default: $CONFIG_VERSION"
else
    CONFIG_VERSION=$p1
fi
PKG_NAME=yum
git config --global user.email "you@example.com"
git config --global user.name "Your Name"
git config --global color.ui false
DIR_NAME=~/ROCm-$CONFIG_VERSION
mkdir $DIR_NAME
if [[ $? -ne 0 ]] ; then
	echo "Directory is already there. Verify it is deleted or renamed before continue." ; exit 1
fi
pushd  $DIR_NAME
mkdir -p ~/bin/
echo "install repo..."
$PKG_NAME  install curl -y && curl https://storage.googleapis.com/git-repo-downloads/repo > ~/bin/repo
chmod a+x ~/bin/repo
echo "repo init..."
~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-$CONFIG_VERSION.x
echo "repo sync..."
~/bin/repo sync
echo "ROCm source is downloaded to $DIR_NAME"
echo "push $DIR_NAME to get there..."
popd

p1=$1
SUDO=sudo
CONFIG_VERSION=4.1
if [[ -z $p1 ]] ; then
    echo "Version not specified. Setting to default: $CONFIG_VERSION"
else
    CONFIG_VERSION=$p1
fi
OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
echo "OS_NAME: $OS_NAME"
case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      ln -s /usr/bin/python3  /usr/bin/python
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

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
$SUDO $PKG_EXEC install curl -y && $SUDO curl https://storage.googleapis.com/git-repo-downloads/repo | $SUDO tee ~/bin/repo
$SUDO chmod a+x ~/bin/repo
echo "repo init..."
$SUDO ~/bin/repo init -u https://github.com/RadeonOpenCompute/ROCm.git -b roc-$CONFIG_VERSION.x
echo "repo sync..."
$SUDO ~/bin/repo sync
echo "ROCm source is downloaded to $DIR_NAME"
echo "push $DIR_NAME to get there..."
popd

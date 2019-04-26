DIR_YETI_ENG_BUNDLE=/root/yeti-eng-bundle
DIR_YETI_CONTENT_BUNDLE=~/root/yeti-content-bundle

if [[ ! -d  $DIR_YETI_ENG_BUNDLE ]] ; then
	echo "$DIR_YETI_ENG_BUNDLE does not exist yet, copying from $GIB_DROP_ROOT/test-apps/yeti..."
	cp -vr $GIB_DROP_ROOT/test-apps/yeti/$DIR_YETI_ENG_BUNDLE ~
else
	echo "$DIR_YETI_ENG_BUNDLE already exist, skipping copy..."
fi

if [[ ! -d  $DIR_YETI_CONTENT_BUNDLE ]] ; then
        echo "$DIR_YETI_CONTENT_BUNDLE does not exist yet, copying from $GIB_DROP_ROOT/test-apps/yeti..."
        cp -vr $GIB_DROP_ROOT/test-apps/yeti/$DIR_YETI_CONTENT_BUNDLE ~
else
        echo "$DIR_YETI_CONTENT_BUNDLE already exist, skipping copy..."
fi


echo "Setup logging (Needed for streaming configurations only – but do it now, so you don't forget):"
sudo mkdir -p /usr/local/cloudcast/log
sudo chmod -R a+rw /usr/local/cloudcast/
sudo ln -s ~/yeti-eng-bundle/lib/ /usr/local/cloudcast/lib
sudo mkdir /log
sudo chmod a+rw /log

sudo apt-get install freeglut3 pulseaudio libpulse-dev

sudo mkdir -p /opt/cloudcast/lib
sudo ln -s /opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so /opt/cloudcast/lib/amdvlk64.so
sudo mkdir -p /usr/local/cloudcast/
sudo ln -s ~/yeti-eng-bundle/lib /usr/local/cloudcast/lib
mkdir -p ~/.local/share/vulkan/icd.d
cp ~/yeti-eng-bundle/etc/vulkan/icd.d/yetivlk.json ~/.local/share/vulkan/icd.d/
sudo mkdir -p /usr/local/cloudcast/etc/yetivlk
sudo cp ~/yeti-eng-bundle/etc/yetivlk/config.json /usr/local/cloudcast/etc/yetivlk

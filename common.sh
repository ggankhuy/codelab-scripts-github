function common_setup () {
	clear
	echo Setup Yeti system for 3dmark on ubuntu 1604 / 1803...
	
	DIR_YETI_ENG_BUNDLE=yeti-eng-bundle
	DIR_YETI_CONTENT_BUNDLE=yeti-content-bundle
	
	if [[ -z $GIB_DROP_ROOT ]] ; then
        	echo "GIB_DROP_ROOT is not defined. Please defined the root in ~/.bashrc"
        	exit 1
	fi

	if [[ ! -d ~/doom/yeti-release/ ]] ; then
		echo "~/doorm/pre-release is not created, creating now." 

		if [[ ~! -d $GIB_DROP_ROOT/test-apps/Doom_Linux/ ]] ; then
			echo "Can not find DOOM source directory. Can not continue setup..."
			exit 1
			
		mkdir -p ~/doom/yeti-release/

		echo "Copying doom now to ~/doom/yeti-release/
		cp -vr $GIB_DROP_ROOT/test-apps/Doom_Linux/* ~/doom/yeti-release/
	
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
	mkdir -p /usr/local/cloudcast/log
	chmod -R a+rw /usr/local/cloudcast/
	ln -s ~/yeti-eng-bundle/lib/ /usr/local/cloudcast/lib
	mkdir /log
	chmod a+rw /log
	
	apt-get install freeglut3 pulseaudio libpulse-dev
	
	mkdir -p /opt/cloudcast/lib
	ln -s /opt/amdgpu-pro/lib/x86_64-linux-gnu/amdvlk64.so /opt/cloudcast/lib/amdvlk64.so
	mkdir -p /usr/local/cloudcast/
	ln -s ~/yeti-eng-bundle/lib /usr/local/cloudcast/lib
	mkdir -p ~/.local/share/vulkan/icd.d
	cp ~/yeti-eng-bundle/etc/vulkan/icd.d/yetivlk.json ~/.local/share/vulkan/icd.d/
	mkdir -p /usr/local/cloudcast/etc/yetivlk
	cp ~/yeti-eng-bundle/etc/yetivlk/config.json /usr/local/cloudcast/etc/yetivlk
	
}

wget https://download.qt.io/official_releases/qt/5.9/5.9.6/qt-opensource-linux-x64-5.9.6.run

if [[ $? -ne 0 ]] ; then
	echo "failed to download qt" ; exit 1
fi


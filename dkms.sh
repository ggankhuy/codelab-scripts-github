clear
apt install -y dh-autoreconf dkms
echo ================================================
echo "Verify prebuild.sh is 755-d on /usr/src/$1/ or /usr/src/<driver_name>"
echo ================================================
sleep 5

if [[ -z $1 ]] || [[ -z $2 ]] ; then
        echo p1 module name or p2 version is not specified.
        echo p1, module: $1
        echo p2, version: $2
else
        sudo dkms uninstall -m $1 -v $2
        sudo dkms remove -m $1 -v $2 --all
        sudo dkms build -m $1 -v $2
        sudo dkms install --force --verbose -m $1 -v $2 -k $(uname -r)
fi






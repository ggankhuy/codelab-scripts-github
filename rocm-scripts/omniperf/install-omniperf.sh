
set -x 

if [[ ! -d  omniperf ]] ; then
    git clone https://github.com/AMDResearch/omniperf.git omniperf
else
    echo "omniperf directory already exist. Assuming checkedout ..."
fi
cd omniperf
export INSTALL_DIR=/opt/omniperf
echo "INSTALL_DIR: $INSTALL_DIR"
python3 -m pip install -t ${INSTALL_DIR}/python-libs -r requirements.txt
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DPYTHON_DEPS=${INSTALL_DIR}/python-libs   -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles ..
if [[ $? -ne  0 ]] ; then
    echo "Error during cmake stage..."
    exit 1
fi
make -j`nproc`

if [[ $? -ne 0 ]]; then
    echo "Error during make stage..."
    exit 1
fi

make install

for i in pymongo astunparse tabulate dash ; do
    echo  "Installing $i..."
    pip3 install $i;
done

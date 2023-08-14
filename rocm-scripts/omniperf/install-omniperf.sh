set -x 
git clone https://github.com/AMDResearch/omniperf.git omniperf
cd omniperf
export INSTALL_DIR=/opt/omniperf
python3 -m pip install -t ${INSTALL_DIR}/python-libs -r requirements.txt
mkdir build
cd build
#cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DPYTHON_DEPS=${INSTALL_DIR}/python-libs   -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles ..
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


git clone ssh://gerritgit/brahma/ec/umr 
if [[ $? -ne 0 ]] ; then 
    echo "unable to clone umr source"
    exit 1
fi
cd umr
yum install -y SDL2 SDL2-devel nanomsg nanomsg-devel llvm llvm-devel libdrm-devel pkg-config cmake libpciaccess-devel zlib-devel ncurses-libs
mkdir build ; cd build
cmake 2>&1 | tee build.log
make -j`nproc` 2>&1 | tee -a build.log

echo "mkdir git.co directory..."
echo -e "amd1234\n" | sudo mkdir /git.co
uname -r

echo "git clone..."
#cd /git.co ; echo -e "g00db0y\n" | sudo git clone ssh://ixt-rack-85@10.216.64.102:32029/home/ixt-rack-85/gg-git-repo/
cd /git.co ; sudo git clone http://gitlab1.amd.com/ggamd000/ad-hoc-scripts.git

echo "checkout dev branch and run yeti setup..."
#cd /git.co/gg-git-repo 
cd /git.co/ad-hoc-scripts

sudo git checkout dev
./yeti-game-test.sh setup
./yeti-game-test.sh doom yeti 2 nolaunch 
./yeti-game-test.sh tr2 yeti 2 nolaunch 
./yeti-game-test.sh quail yeti 2 nolaunch 
./yeti-game-test.sh 3dmark yeti 2 nolaunch 


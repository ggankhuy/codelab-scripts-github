counter=0
yum groupinstall "Development Tools" -y | tee setup.development.tools.log
pip3 install --upgrade pip 2>&1 | tee pip.install.upgrade.log
for i in \
"python3 -m pip3 install --upgrade pip" \
"yum update -y" \
"yum install zlib python-devel -y"
do
    echo "------------------------------" | tee setup.$counter.log
    #echo "DBG: executing '$i'..." | tee -a setup.$counter.log
    echo "------------------------------"  | tee -a setup.$counter.log
    #$i 2>&1 | tee -a setup.$counter.log
    counter=$((counter+1))
done 

counter=0
for i in neuralnet matplotlib pandas sklearn Pillow scipy==1.2.0
do
    echo "------------------------------" | tee setup.pip3.$counter.log
    echo "DBG: executing '$i'..." | tee -a setup.pip3.$counter.log
    echo "------------------------------"  | tee -a setup.pip3.$counter.log
    pip3 install $i 2>&1 | tee -a setup.pip3.$counter.log
    counter=$((counter+1))
done

#pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/rocm4.5.2"  2>&1 | tee setup.torch.torchvision.log

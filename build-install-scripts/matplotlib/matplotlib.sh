# worked on centos GUI. Python must be reinstalled afterward. (3.9)

# 1050  pip3 list | grep matplot
# 1058  yum install -y python3-tkinter
# 1060  yum install -y python3-tkinter
# 1062  pip3 install pyqt5 
# 1064  yum install -y tk-devel

LOG_FOLDER=./log

for i in python3-tkinter; do
    echo "Installing $i..." 2>&1 | tee -a $LOG_FOLDER/matplotlib.log
    yum install -y $i  2>&1 | tee -a $LOG_FOLDER/matplotlib.log
done

for i in pyqt5 ; do
    echo "pip3 install $i..." 2>&1 | tee -a $LOG_FOLDER/matplotlib.log
    pip3 install $i 2>&1 | tee -a $LOG_FOLDER/matplotlib.log
done
    




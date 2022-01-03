echo "set tabsize 4" | sudo tee  ~/.nanorc
echo "set tabstospaces " | sudo tee -a ~/.nanorc
echo "set tabsize 4" | sudo tee /root/.nanorc
echo "set tabstospaces " | sudo tee -a /root/.nanorc

echo "$user home nanorc: "
sudo cat ~/.nanorc
echo "root home nanorc: "
sudo cat /root/.nanorc

OS_NAME=`cat /etc/os-release  | grep ^NAME=  | tr -s ' ' | cut -d '"' -f2`
echo "OS_NAME: $OS_NAME"
case "$OS_NAME" in
   "Ubuntu")
      echo "Ubuntu is detected..."
      PKG_EXEC=apt
      ;;
   "CentOS Linux")
      echo "CentOS is detected..."
      PKG_EXEC=yum
      ;;
   *)
     echo "Unsupported O/S, exiting..." ; exit 1
     ;;
esac

sudo $PKG_EXEC install git -y
sudo git config --global credential.helper store
sudo git config --global user.email "you@example.com"
sudo git config --global user.name "Your Name"
sudo cat ~/.gitconfig | sudo tee /root/.gitconfig


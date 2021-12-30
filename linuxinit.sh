echo "set tabsize 4" > ~/.nanorc
echo "set tabstospaces " >> ~/.nanorc
echo "set tabsize 4" > /root/.nanorc
echo "set tabstospaces " >> /root/.nanorc

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

$PKG_EXEC install git -y
git config --global credential.helper store

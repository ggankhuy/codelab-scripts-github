mv ~/.ssh/config ~/.ssh/config.bak
echo ~/.ssh/config is backed up as ~/.ssh/config.bak if it existed.
cp gerritt/gerrit_ssh_config ~/.ssh/config

mv ~/.gitconfig ~/.gitconfig.bak
echo ~/.gitconfig is backed up as  ~/.gitconfig.bak  if it existed.
cp gerrit_gerrit_git_cofnig ~/.gitconfig

echo "generating 4k size ssk key, keep pressing without inputting any password or passphrase...
ssh-keygen -b 4096
echo "paste it into http://gerrit-git.amd.com/settings/"
cat ~/.ssh/id_rsa.pub


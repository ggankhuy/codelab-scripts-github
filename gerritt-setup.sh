SSH_CONFIG=~/.ssh/config
GIT_CONFIG=~/.gitconfig
DATE=`date +%Y%m%d-%H-%M-%S`

if [[ -f $SSH_CONFIG ]] ; then
	mv $SSH_CONFIG $SSH_CONFIG.$DATE.bak
	echo $SSH_CONFIG is backed up as $SSH_CONFIG.$DATE.bak if it existed.
fi
cp gerritt/gerrit_ssh_config $SSH_CONFIG

if [[ -f $GIT_CONFIG ]] ; then
	mv $GIT_CONFIG $GIT_CONFIG.$DATE.bak
	echo $GIT_CONFIG is backed up as  $GIT_CONFIG.$DATE.bak  if it existed.
fi
cp gerritt/gerrit_git_config $GIT_CONFIG

echo "generating 4k size ssk key, keep pressing without inputting any password or passphrase..."
ssh-keygen -b 4096
echo "paste it into http://gerrit-git.amd.com/settings/"
cat ~/.ssh/id_rsa.pub


set -x 
echo "Verify you run this test after pasting pubkey to gerrit-git.amd.com!!!"
ssh gerritgitmaster gerrit version

echo "Following should run after about 15min after setup in order for the key to allowe propagate..."
echo "sleeping for 15 min or come back after 15min!"
sleep $((15*60))
ssh gerritgit gerrit version


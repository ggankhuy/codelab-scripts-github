set -x
wget -qO /usr/local/bin/ninja.gz https://github.com/ninja-build/ninja/releases/latest/download/ninja-linux.zip
gunzip /usr/local/bin/ninja.gz
chmod a+x /usr/local/bin/ninja

#get ninjatracing

NINJA_TRACING_PATH=~/ninjatracing
ln -s /usr/bin/python3 /usr/bin/python
git clone https://github.com/nico/ninjatracing.git $NINJA_TRACING_PATH
ln -s $NINJA_TRACING_PATH/ninjatracing /usr/bin


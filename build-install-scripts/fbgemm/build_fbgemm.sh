env_name=`echo $CONDA_DEFAULT_ENV | awk "{print $2}"`

if [[ -z $env_name ]] ; then
    echo "Make sure conda environment is activated"
    exit 1
else
    echo "env_name: $env_name"
fi

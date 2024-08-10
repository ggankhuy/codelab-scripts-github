# p1 - env varname
# p2 - env value.
# example: export_bashrc ROCM_DIR /opt/rocm
# example: export_bashrc ROCM_DIR $ROCM_DIR (if ROCM_DIR is defined)

function export_bashrc() {
    env_name=$1
    env_value=$2

    export $env_name=$env_value

    if [[ -z $env_name ]] || [[ -z env_value ]] ; then echo "env_name or env_value is empty" ; return 1; fi

    [[ `grep "export.*$env_name" ~/.bashrc` ]] && \
    sed -i --expression "s/export.*${env_name}.*/export ${env_name}=${env_value}/g" ~/.bashrc || \
    echo "export $env_name=$env_value" | tee -a ~/.bashrc

}

function export_bashrc_delim_alt() {
    env_name=$1
    env_value=$2

    export $env_name=$env_value

    if [[ -z $env_name ]] || [[ -z env_value ]] ; then echo "env_name or env_value is empty" ; return 1; fi

    [[ `grep "export.*$env_name" ~/.bashrc` ]] && \
    sed -i --expression "s@export.*${env_name}.*@export ${env_name}=${env_value}@g" ~/.bashrc || \
    echo "export $env_name=$env_value" | tee -a ~/.bashrc
}


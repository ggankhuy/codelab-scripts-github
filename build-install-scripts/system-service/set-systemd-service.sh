set -x

function usage() {
    set +x
    echo "$0    --help : to display this help."
    echo "$0    --init-script=path_to_init_script"
    echo "      --name=<services_name>"
    echo "      --type=<type of startup startup script/exec used in startup service>"
    echo "      value for --type: "
    echo "      sh or shell for shell script"
    echo "      py, pyt or python for python"
    echo "      py3, pyt3 or python3 for python3 to specify python3 explicitly"
    echo "      --desc=<description of startup script/service>"
    echo "Example: $0 \
        --init-script=/usr/bin/my.sh \
        --name=myserv \
        --type=shell \
        --desc='mystartup script' : \
        will setup myserv that will execute my.sh during boot"
}
    
for var in "$@"
do
    echo var: $var
    case "$var" in
        *--help*)
            usage
            exit 0
            ;;
        *--init-script=*)
            p_script=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *--name=*)
            p_name=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *--type=*)
            p_type=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *--desc=*)
            p_desc=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *)
            echo "Unknown cmdline parameter: $var"
            usage
            exit 1
            ;;
    esac
done

systemctl stop firewalld.service
PYTHON_PATH=/usr/bin/python
PYTHON3_PATH=/usr/bin/python3
BASH_PATH=/bin/bash

if [[ ! -f $PYTHON_PATH ]] ; then
    PYTHON_PATH=`which python`
fi

if [[ ! -f $PATH_PATH3 ]] ; then
    PYTHON_PATH3=`which python3`
fi

if [[ ! -f $BASH_PATH ]] ; then
    BASH_PATH=`which bash`
fi

case "$p_type" in
    shell)
        if [[ -z $BASH_PATH ]] ; then
            echo "bash is not found. Unable to set shebang at the beginning. Services initialization therefore likely fail although setup would correctly"
        else
            SHEBANG="#!/$BASH_PATH"
        fi
        ;;
    python)
        if [[ -z $PYTHON_PATH ]] ; then
            echo "python path is not found. Unable to set shebang at the beginning. Services initialization therefore likely fail although setup would correctly"
        else
            SHEBANG="#!/$PYTHON_PATH"
        fi
        ;;

    python3)
        if [[ -z $PYTHON3_PATH ]] ; then
            echo "python3 path: is not found. Unable to set shebang at the beginning. Services initialization therefore likely fail although setup would correctly"
        else
            SHEBANG="#!/$PYTHON3_PATH"
        fi
        ;;
    *)
        echo "Invalid type: supported types are so far 'shell' or 'python'"
        usage
        exit 1
        ;;
esac

if [[ -z $p_script ]] || [[ -z $p_name ]] ; then
    echo "Both of --init-script and --name must be specified and mandatory."
    usage 
    exit 1
fi

SERVICES_PATH_SOFT=/etc/systemd/system/multi-user.target.wants/$p_name.service
SERVICES_PATH=/usr/lib/systemd/system/$p_name.service
sudo ln -s $SERVICE_PATH_SOFT $SERVICE_PATH

if [[ -f $SERVICES_PATH ]] ; then
    echo "Service already exists: $SERVICE_PATH"
    exit 1
fi

if [[ -z $p_desc ]] ; then
    echo "Description of the startup service was not specified. It is strongly encouraged to do so!"
    echo "After setup, you can add in following file:  /$SERVICES_PATH"
fi

echo "$SHEBANG" | sudo tee tmp.shebang
cat $p_script | sudo tee tmp.script

cat tmp.shebang | sudo tee $p_script
cat tmp.script | sudo tee -a $p_script

cat template.service | sudo tee $SERVICES_PATH
chmod 755 $p_script
filename=`basename $p_script`

if [[ -f /usr/bin/$filename ]] ; then echo "/usr/bin/$filename already exist" ; exit 1 ; fi

cp $p_script /usr/bin/$filename
mv tmp.script $p_script
rm -rf tmp.*
p_script_new=/usr/bin/$filename
sudo sed -i "/ExecStart/c \ \ExecStart = $p_script_new" $SERVICES_PATH

if [[ $p_desc ]] ; then
    sudo sed -i "/Description/c \ \Description = $p_desc" $SERVICES_PATH
fi
sudo systemctl enable $p_name.service
sudo systemctl start $p_name.service
sudo systemctl status $p_name.service


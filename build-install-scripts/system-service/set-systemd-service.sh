set -x

function usage() {
    echo "$0 --help : to display this help."
    echo "$0 --init-script=<path_to_init_script> --name=<services_name>"
    echo "Example: $i --init-script=/usr/bin/my.sh --name=myserv : will setup myserv that will execute my.sh during boot"
}
    
for var in "$@"
do
    echo var: $var
    case "$var" in
        *--help*)
            usage
            exit 0
            ;;
        *--type=*)
            p_type=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *--init-script=*)
            p_script=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *--name=*)
            p_name=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *--decs=*)
            p_desc=`echo $var | awk -F '=' '{print $2}'`
            ;;
        *)
            echo "Unknown cmdline parameter: $var"
            usage
            exit 1
            ;;
    esac
done
PYTHON_PATH=/usr/bin/python
PYTHON3_PATH=/usr/bin/python3
BASH_PATH=/bin/bash

if [[ ! -f $PYTHON_PATH ]] ; then
    PYTHON_PATH=`which python`
fi

if [[ ! -f $PATH_PATH3 ]] ; then
    PYTHON_PATH3=`which python3`
fi

if [[ z -f $BASH_PATH ]]; then
    BASH_PATH=`which bash`
fi

case "$p_type" in
    sh)
    shell)
        if [[ -z $BASH_PATH ]] ; then
            echo "bash is not found. Unable to set shebang at the beginning. Services initialization therefore likely fail although setup would correctly"
        else
            SHEBANG="#!/$BASH_PATH"
        fi
        ;;
    py)
    pyt)
    python)
        if [[ -z $PYTHON_PATH ]] ; then
            echo "python path is not found. Unable to set shebang at the beginning. Services initialization therefore likely fail although setup would correctly"
        else
            SHEBANG="#!/$PYTHON_PATH"
        fi
        ;;

    py3)
    pyt3)
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
    clear
    echo "Both of --init-script and --name must be specified and mandatory."
    usage 
    exit 1
fi

SERVICES_PATH=/etc/systemd/system/multi-user.target.wants/$p_name.service

if [[ -z $p_desc ]] ; then
    echo "Description of the startup service was not specified. It is strongly encouraged to do so!"
    echo "After setup, you can add in following file:  /$SERVICES_PATH"
fi

sed -i '1s/^/$SHEBANG\n/' $p_script
cat template.service | sudo tee -a /etc/systemd/system/multi-user.target.wants/$p_name.service
sudo sed -i "/ExecStart/c \ \ExecStart = $p_script" $SERVICES_PATH

if [[ $p_desc ]] ; then
    sudo sed -i "/Description/c \ \Description = $p_desc" $SERVICES_PATH
fi

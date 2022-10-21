function r1() {
    p1=$1
    echo p1: $1
    p1=$((p1-1))

    if [[ $p1 -ge 0 ]] ; then
        r1 $p1
    else
        return 0
    fi
}

r1 10

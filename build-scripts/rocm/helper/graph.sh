p1=$1
echo "building $p1..."

components=()

function f1() {
    component=$1
    echo "searching for $component..."

    while IFS= read -r line; do
        echo $line
        if [[ ! -z `echo $line | grep "$component"` ]] ; then
            echo "found..."

            # gather dependencies...

            dependency_line=`echo $line | cut -d ':' -f2`        
            echo "dependency_line: $dependency_line"

            # for each dependencies call f1 recursively.

            break
        fi
    done < graph.dat
}


f1 rocblas

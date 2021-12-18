#declare -a arr=("asd asd" "aa11" "1133 asd")
arr=("asd asd" "aa11" "1133 asd")
for i in "${arr[@]}"
do
    echo $i
done

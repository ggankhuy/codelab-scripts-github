rm -rf ../*.deb
rm -rf ./log/*
rm -rf ./binary/*
for i in 3.8 3.9 4.0 4.1 ; do
        echo $i
        mkdir -p log/
        git checkout roc-$i.x
        make clean
        make -j8 bindeb-pkg | tee log/build.$i.log
        mkdir -p binary/$i
        mv ../*.deb ./binary/$i
done


rocm_path=/opt/rocm
rocm_rpath=" -Wl,--enable-new-dtags -Wl,--rpath,/opt/rocm/lib:/opt/rocm/lib64"
compiler="hipcc"
cmake_executable="cmake"
project_name=vector

LOG_FILE_BUILD=cmake-stdout.log
LOG_FILE_EXEC=exec.log

#for sub_project_name in vector vector-4 ; do
rm -rf bindir/*

for sub_project_name in vector vector4 vector1024 ; do
    echo "Generating project: $project_name..."
    BUILD_DIR=build-$sub_project_name
    mkdir ./log
    mkdir $BUILD_DIR
    mkdir ./bindir
    #cd $BUILD_DIR
    #rm -rf ./*

    export PROJECT_NAME=$sub_project_name

    #hipcc --save-temps -DSUB_PROJECT_NAME=$sub_project_name $FILENAME.cpp -o 
    echo hipcc -DOPT_SUB_PROJECT_NAME=$sub_project_name $project_name.cpp -o $project_name
    sleep 4
    hipcc -DOPT_SUB_PROJECT_NAME=$sub_project_name $project_name.cpp -o $project_name

    mv ${project_name} ./bindir/${sub_project_name}
    ./bindir/${project_name} | tee -a ./$BUILD_DIR/$LOG_FILE_EXEC

done

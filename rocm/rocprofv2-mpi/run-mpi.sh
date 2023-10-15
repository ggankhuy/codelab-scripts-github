APP=p61
sudo hipcc $APP.cpp -o $APP.out
OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun -n 4  ./$APP.out
 

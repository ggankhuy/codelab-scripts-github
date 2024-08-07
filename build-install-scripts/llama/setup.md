1) Login either as root or non-root user. Do not run as sudo.
2) If login as non-root user, the package file containing wheels and parent directory should be owned same non-root account, otherwise
will cause permission issue. i.e. change ownership of parent directory containing setup1-llama.sh script.
cd ..
for i in chown chgrp ; do $i <non-root username> <dirname> ; done
3) run setup-llama2-stg1.sh --env_name=<name_of_conda_env> --pkg_name=<name of tar file w/o extension>
example: setup-llama2-stg1.sh --env_name=llama2-test --pkg_name=20240719_quanta_llama2_rocm_6.2.0-8
4) re-login (to make bashrc/new conda environment settings take effect)
5) verify new conda precedes the shell prompt implying successful activation:
example: (<name_of_conda_env>) [<USER>@<HOSTNAME> llama]$
6) run setup-llama2-stg2.sh (will install wheels, build magma, setup paths)
7) re-login (to make new bashrc environment setting take effect)
8) a) option 1. run run_llama2_70b_bf16.sh (from inside 20240719_quanta_llama2_rocm_6.2.0-8 directory untarred)
   b) option 2. run run-llama2.sh (wrapper which calls run_llama2_70b_bf16.sh and generates logs int ./log/<DATE_TOKEN> directory)

1) run setup-llama2-stg1.sh --env_name=<name_of_conda_env> --pkg_name=<name of tar file w/o extension>
example: setup-llama2-stg1.sh --env_name=llama2-test --pkg_name=20240719_quanta_llama2_rocm_6.2.0-8
2) re-login (to make bashrc/new conda environment settings take effect)
3) verify new conda precedes the shell prompt implying successful activation: 
example: (llama2) [<USER>@<HOSTNAME> llama]$
4) run setup-llama2-stg2.sh (will install wheels, build magma, setup paths)
5) re-login (to make new bashrc environment setting take effect)
6) a) option 1. run run_llama2_70b_bf16.sh (from inside 20240719_quanta_llama2_rocm_6.2.0-8)
   b) option 2. run run-llama2.sh (wrapper which calls run_llama2_70b_bf16.sh and generates logs int ./log/<DATE_TOKEN> directory)


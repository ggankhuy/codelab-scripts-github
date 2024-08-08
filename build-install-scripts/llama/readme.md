1) Run as either root or nonroot account. Do not run as sudo, running as sudo is untested.
1a) If running as non root account, change parent directory to be owned by non root account's user:
cd .. (go up one level to parent dir)
for i in chgrp chown ; do sudo $i <USER> <child_dir> ; done
done
cd <child_dir>

2) ./setup-llama2-stg1.sh --env_name=<conda_env_name> --pkg_name=<pkg_name without extension>
example: ././setup-llama2-stg1.sh --env_name=llama2-test --pkg_name=20240719_quanta_llama2_rocm_6.2.0-8
3) re-login (so that conda environment will take effect)
4) ./setup-llama2-stg2.sh (will install wheels, build magma, setup PATHs etc.,
5) re-login again (so that environment setting will take effect)
6) option #1 from --pkg_name directory run: run_llama2_70b_bf16.sh
   option #2 run wrapper instead: run-llama2.sh (will run same as option #1, will additioanlly create logs in ./log/<DATE_TOKEN> directory.


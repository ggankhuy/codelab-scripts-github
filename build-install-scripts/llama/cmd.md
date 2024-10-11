export ENV_NAME=dev-tmp-1-torch-only; export STG=stg1 ;./setup-llama2-$STG.sh --env_name=$ENV_NAME --pkg_name=20240719_quanta_llama2_rocm_6.2.0-8 2>&1 | tee log/env.$ENV_NAME.$STG.log

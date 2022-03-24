# For this script to work, p297.py must be run with option set to: 
# CONFIG_SAVE_MODEL_MODE=CONFIG_SAVE_MODEL_MODE_SAVED_MODEL
# resulting directory structure should be:
# root@nonroot-Standard-PC-i440FX-PIIX-1996:~/dev-learn/gpu/tflow/tensorflow/tflow-2nded# tree p297
# p297
#├── 0001
#│   ├── assets
#│   ├── saved_model.pb
#│   └── variables
#│       ├── variables.data-00000-of-00001
#│       └── variables.index
#├── assets
#├── keras_metadata.pb
#├── saved_model.pb
#└── variables
#    ├── variables.data-00000-of-00001
#    └── variables.index
# 5 directories, 7 files
 
TEST_MODE=0

if [[ $TEST_MODE -eq 1 ]] ; then
    echo "TEST MODE..."
    sleep 3
    saved_model_cli run --dir p297/0001 --tag_set serve --signature_def serving_default --inputs flatten_input=test.npy
else
    echo "NON TEST MODE..."
    MODEL_NAME=p297
    docker pull tensorflow/serving
    docker run -it --rm -p 8500:8500 -p 8501:8501 \
        -v "$MODEL_NAME:/models/$MODEL_NAME" \
        -e MODEL_NAME=$MODEL_NAME \
        tensorflow/serving
fi

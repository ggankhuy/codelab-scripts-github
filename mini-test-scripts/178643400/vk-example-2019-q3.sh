#!/bin/bash

run_module=${1:-all}

[[ "$VKEXAMPLE_RT_LOG" ]] && VKEXAMPLE_RUNTIME_LOG="$VKEXAMPLE_RT_LOG" || \
    VKEXAMPLE_RUNTIME_LOG="/work/test-report/vk-example.$run_module.log"
[[ "$VKEXAMPLE_RES_LOG" ]] && Xcommon_report="$VKEXAMPLE_RES_LOG"

source $(dirname $0)/xcommon.sh
#func_restrict_parameter "$run_module" "default" "mini"
func_check_environment
func_check_amdgpu

mkdir -p $(dirname $VKEXAMPLE_RUNTIME_LOG)

os_id=$(awk -F '=' '/^ID=/ {print $NF;}' /etc/os-release)
cd $(dirname $0)/../../
YETI_HOME=$PWD/test-apps/yeti
YETI_CONTENT_HOME=$YETI_HOME/yeti-content-bundle
YETI_ENGINE_HOME=$YETI_HOME/yeti-eng-bundle
YETI_ENGINE_HOME=$YETI_HOME/ggp-eng-bundle
YETI_VK_EXAMPLE_HOME=$PWD/test-apps/*vk_example*

if [ "$os_id" == "debian" ];then # G for debian
    export LD_LIBRARY_PATH=$YETI_ENGINE_HOME/lib:$LD_LIBRARY_PATH
    export VK_LOADER_DISABLE_YETI_EXT_WHITELIST=1
    export DISABLE_LAYER_GOOGLE_YETI=1
    source $YETI_ENGINE_HOME/env/vce_nostreamer.sh# root permission
    VULKAN_LAYER_FILE="/etc/vulkan/implicit_layer.d/yeti_vulkan_layer.json"

    [[ ! -d /etc/vulkan ]] && mkdir -p /etc/vulkan
    [[ ! -e $(dirname $VULKAN_LAYER_FILE) ]] && ln -s $YETI_ENGINE_HOME/$(dirname $VULKAN_LAYER_FILE) /etc/vulkan/
    [[ ! -e $VULKAN_LAYER_FILE ]] && cp $YETI_ENGINE_HOME/$VULKAN_LAYER_FILE $VULKAN_LAYER_FILE
    [[ ! -d /usr/local/cloudcast/log ]] && mkdir -p /usr/local/cloudcast/log
    [[ ! -e /usr/local/cloudcast/lib ]] && ln -s $YETI_ENGINE_HOME/lib /usr/local/cloudcast/lib
    #cd $(dirname $(find test-apps/*vk_example* -name TestExecutor|grep -v gibraltar |head -n 1))
    cd $(dirname $(find test-apps/*vk_example* -name TestExecutor | head -n 1))
    ./TestExecutor --offscreen 2>&1 1>$VKEXAMPLE_RUNTIME_LOG
elif [ "$os_id" == "ubuntu" ];then # Mainline for ubuntu
    cd $(dirname $(find test-apps/*vk_example* -name TestExecutor|grep -v gibraltar |head -n 1))
    ./TestExecutor --offscreen --disable-extension-secure-strings 2>&1 1>$VKEXAMPLE_RUNTIME_LOG
fi

if [ "$(grep '^\[' $VKEXAMPLE_RUNTIME_LOG| grep -vi 'success' |grep -vi 'FAIL:0')" ];then
    func_write_message "$(grep '^\[' $VKEXAMPLE_RUNTIME_LOG| grep -vi 'success' |grep -vi 'FAIL:0')"
    func_script_quit 1 "VK Example run $run_module Catch failed case"
fi

if [ "$os_id" == "debian" ];then
    rm -rf /usr/local/cloudcast/log
    rm -rf /usr/local/cloudcast/lib
    rm -rf $VULKAN_LAYER_FILE
    rm -rf $YETI_VK_EXAMPLE_RT

    unset LD_LIBRARY_PATH
    unset VK_LOADER_DISABLE_YETI_EXT_WHITELIST
    unset DISABLE_LAYER_GOOGLE_YETI
elif [ "$os_id" == "ubuntu" ];then
    echo "skip"
fi

func_script_quit

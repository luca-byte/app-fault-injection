#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"

mkdir -p ${global_PWD}/${DIR}


echo ${DIR}
export LOG_DIR=${DIR}

bash ${global_PWD}/APP_Fault_injections/bash/crbq/Neurons_cfg_FI.sh $start_layer $stop_layer ${DIR} 


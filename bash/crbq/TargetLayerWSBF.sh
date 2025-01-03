#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"


mkdir -p ${global_PWD}/${DIR}

input_args=(0 2 4)

array_size=${#input_args[@]}

for ((i=0; i<$array_size; i++)); do
    bash ${global_PWD}/APP_Fault_injections/bash/crbq/Weight_cfg_FI.sh ${input_args[$((i))]} ${DIR}
done

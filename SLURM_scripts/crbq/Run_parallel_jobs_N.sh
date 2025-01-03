#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"

mkdir -p ${global_PWD}/${DIR}


echo ${DIR}
export LOG_DIR=${DIR}

for ((i=0; i<$array_size; i++)); do
    sbatch --output=$DIR/cnf${input_args[$((i))]}_lyr${start_layer}_stdo_%A_%a.log --error=$DIR/cnf${input_args[$((i))]}_lyr${start_layer}_stde_%A_%a.log ${global_PWD}/APP_Fault_injections/bash/crbq/Neurons_cfg_FI.sh $start_layer $stop_layer ${DIR} 
done

#!/bin/bash

# 1 Activate the virtual environment
# conda init
# conda deactivate

# cd  ~/Desktop/Ph.D_/projects/APP_FI/code

# conda activate APP_FSIM


PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id=0

start_layer="$1"
stop_layer="$2"
DIR="$3"

Sim_dir=${global_PWD}/${DIR}/lyr${start_layer}-${stop_layer}_JOBID${job_id}_N
mkdir -p ${Sim_dir}

cp ${global_PWD}/APP_Fault_injections/configs/cifar10/teacher/mnasnet.yaml ${Sim_dir}
cp ${global_PWD}/APP_Fault_injections/configs/cifar10/teacher/Fault_descriptor.yaml ${Sim_dir}
sed -i "s/layers: \[.*\]/layers: \[$start_layer,$stop_layer\]/" ${Sim_dir}/Fault_descriptor.yaml
sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml

cd ${Sim_dir}

python ${global_PWD}/APP_Fault_injections/script/image_classification_FI_neuron_ber.py \
        --config ${Sim_dir}/mnasnet.yaml\
        --device cuda\
        --log ${Sim_dir}/mnasnet.log\
        --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/lyr${start_layer}_stdo.log 2> ${global_PWD}/${DIR}/lyr${start_layer}_stde.log

echo
echo "All done. Checking results:"

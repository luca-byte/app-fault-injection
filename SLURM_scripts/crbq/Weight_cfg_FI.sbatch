#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=WSBF
#SBATCH --mail-type=ALL
#SBATCH --mail-user=giuseppe.esposito@polito.it


# 1 Activate the virtual environment
source ~/miniconda3/bin/activate
conda deactivate

cd  ~/Desktop/Ph.D_/projects/APP_FI/code/APP_Fault_injections

conda activate APP_FSIM
module load nvidia/cudasdk/11.6

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id=0

target_layer="$1"
DIR="$2"

Sim_dir=${global_PWD}/${DIR}/lyr${target_layer}_JOBID${job_id}_W
mkdir -p ${Sim_dir}

cp ${global_PWD}/APP_Fault_injections/configs/cifar10/teacher/LeNetDrop.yaml ${Sim_dir}
cp ${global_PWD}/APP_Fault_injections/configs/cifar10/teacher/Fault_descriptor.yaml ${Sim_dir}
sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

cd ${Sim_dir}

python ${global_PWD}/APP_Fault_injections/script/image_classification_FI_sbfm.py \
        --config ${Sim_dir}/LeNetDrop.yaml\
        --device cuda\
        --log ${Sim_dir}/log/LenetDrop.log\
        --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/lyr${target_layer}_stdo.log 2> ${global_PWD}/${DIR}/lyr${target_layer}_stde.log



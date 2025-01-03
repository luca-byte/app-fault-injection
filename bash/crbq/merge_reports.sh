#!/bin/bash

source ~/miniconda3/bin/activate APP_FI


PWD=`pwd`
Global_path="$PWD"

folder="$1"
workers="$2"
echo $Global_path

echo $folder
echo $workers
echo ${Global_path}/${folder}

python ${Global_path}/code/APP_Fault_injections/bash/crbq/merge_reports.py --path ${Global_path}/${folder} --workers ${workers}

echo "merge finishied"

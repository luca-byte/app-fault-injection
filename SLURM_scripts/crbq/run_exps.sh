PWD=`pwd`

global_PWD="$PWD"

bash ${global_PWD}/APP_Fault_injections/SLURM_scripts/crbq/Run_parallel_jobs_N.sh FSIM_N_HPC_LenetDrop 0 1 

bash ${global_PWD}/APP_Fault_injections/SLURM_scripts/crbq/Run_parallel_jobs_N.sh FSIM_N_HPC_LenetDrop 1 2

bash ${global_PWD}/APP_Fault_injections/SLURM_scripts/crbq/Run_parallel_jobs_N.sh FSIM_N_HPC_LenetDrop 2 3

bash ${global_PWD}/APP_Fault_injections/SLURM_scripts/crbq/Run_parallel_jobs_N.sh FSIM_N_HPC_LenetDrop 3 4

bash ${global_PWD}/APP_Fault_injections/SLURM_scripts/crbq/Run_parallel_jobs_N.sh FSIM_N_HPC_LenetDrop 4 5

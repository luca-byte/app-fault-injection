# APP_Fault_injections

This is a Fault/error injection framework for reliability evaluation of any DNN architectures including split computing neural networks [sc2-benchmark](https://github.com/yoshitomo-matsubara/sc2-benchmark).

# prerequisites
Install miniconda environmet if you already have it ignore this step
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b
```

# Getting started on a Linux x86\_64 PC
```bash
# APP_Fault_injections
git clone https://github.com/GiuseppeEsposito98/APP_Fault_injections.git
cd APP_Fault_injections
find . -name "*.sh" | xargs chmod +x

# pytorchfi 
git clone https://github.com/GiuseppeEsposito98/extended_pytorchfi.git

# create the conda environmet and install the required dependencies
cp environment.yaml ../environment.yaml
cd ..
conda deactivate

conda env create -f environment.yaml
conda deactivate
source ~/miniconda3/bin/activate APP_FSIM

python -m pip install -e .

python -m pip install -e ./APP_Fault_injections/extended_pytorchfi/
```

# Directory structure (simplified)
```
sc2-benchmark.
             ├── configs
             ├── environment.yaml
             ├── LICENSE
             ├── MANIFEST.in
             ├── Pipfile
             ├── README.md
             ├── SC_Fault_injections
             │   ├── bash
             │   │   ├── Check_DNN_archs.sh
             │   │   ├── crbq
             │   │   │   ├── merge_reports.py
             │   │   │   ├── merge_reports.sh
             │   │   │   ├── Neurons_cfg_FI.sh
             │   │   │   ├── NeuronBER.sh
             │   │   │   ├── TargetLayerWSBF.sh
             │   │   │   └── Weight_cfg_FI.sh
             │   ├── configs
             │   ├── Dataset_script
             │   ├── environment.yaml
             │   ├── Pipfile
             │   ├── extended_pytorchfi
             │   ├── report_analysis
             │   ├── script
             │   └── SLURM_scripts
             │   │   ├── crbq
             │   │   │   ├── merge_reports.py
             │   │   │   ├── merge_reports.sh
             │   │   │   ├── Neurons_cfg_FI.sh
             │   │   │   ├── NeuronBER.sh
             │   │   │   ├── TargetLayerWSBF.sh
             │   │   │   └── Weight_cfg_FI.sh
             ├── script
             ├── setup.cfg
             ├── setup.py
             └── tree.txt
```
# How to use this framework?
1. deactivate the base conda environmet and activate the APP-FSIM environment
2. change in your terminal the directory to the APP_Fault_injections directory.
2. run the Fsim command 
```bash
bash ./SC_Fault_injections/bash/crbq/TargetLayerWSBF.sh FSIM_W 
```
This command will create a folder called FSIM_W and it will star performing fault simulations to the layer 0 2 and 4 on Mnasnet trained and tested on CIFAR10 

**Note** It is recomended to use HPC system to execute several FSIMs in parallel. for that purposes you can follow exactly the same steps but intead to use 
bash scripts use the SLURM scripts 
```bash
bash ./SC_Fault_injections/SLURM_scripts/crbq/Run_parallel_jobs_W.sh FSIM_W 
```

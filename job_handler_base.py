
# This code is going to submit shell scripts
import os
import subprocess
import sys
wait = True
if wait:
    job_seq = 3100974
    hold_modifier = f"-hold_jid {job_seq}"
else:
    hold_modifier = ""
python_file = 'pfm_dataset/src/main.py'
simulation_case = ['tension']
job_array = '1-500:1'
rerun_jobs = True
ds_models = ['amor']
mesh_size_ = 800
comment = "2c"
init_crack_add = "pfm_dataset/initial_cracks_2c"
delta_T = 1e-6
num_steps = 5000
g_c_ = 1/1
e_ = 1000.0e3/1
domain_size = 2.0
# Job spec
num_nodes = 1
num_cores = 16
job_time = '12:00:00'

# log directory
# job script directory
os.makedirs(f'pfm_dataset/jobs', exist_ok=True)

for sim_case_ in simulation_case:
    for model_ in ds_models:
        out_file = f'{sim_case_}_{model_}_{comment}'
        job_name = f'{sim_case_[0]}_{model_[0]}_{comment}'
        os.makedirs(f'pfm_dataset/logs/{job_name}', exist_ok=True)
        job_script = f"""#!/bin/bash -l
#$ -pe mpi_{num_cores}_tasks_per_node {num_cores}
#$ -N {job_name}
#$ -l h_rt={job_time}
# Merge stderr into the stdout file, to reduce clutter.
#$ -j y
#$ -m beas
#$ -o pfm_dataset/logs/{job_name}/
#$ -t {job_array}
## end of qsub options

echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="
export HDF5_USE_FILE_LOCKING="FALSE"
export FFCX_CACHE_DIR="/tmp/fenics_cache_$SGE_TASK_ID"
mkdir -p $FFCX_CACHE_DIR
module load openmpi
module load miniconda
conda activate fenicsx-mpi-env
mpirun -np $NSLOTS python3 {python_file} --sim_case {sim_case_} --delta_T {delta_T} --num_steps {num_steps} --init_crack_add {init_crack_add} --model {model_} --mesh_size {mesh_size_} --prefix {out_file} --job_id $JOB_ID --seed $SGE_TASK_ID --g_c {g_c_} --e_ {e_} --domain_size {domain_size}
# Clean up task-specific cache
rm -rf $FFCX_CACHE_DIR
        """
        job_script_file = f'pfm_dataset/jobs/{job_name}_{sim_case_}_{model_}_{mesh_size_}.sh'
        with open(job_script_file, 'w') as f:
            f.write(job_script)
        os.system(f'qsub {job_script_file}')


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
python_file = 'pfm_lite/src/main_dynamic_claude.py'
# Dynamic script supports tension + Miehe only; these lists survive as labels
# for the output directory naming.
simulation_case = ['tension']
job_array = '1'
rerun_jobs = True
ds_models = ['miehe']
mesh_size_ = 400
comment = "dynamic_big_ramp_long_domain_size_1"
init_crack_add = "pfm_lite/initial_cracks"

# Dynamic-specific parameters (must match CLI args in main_dynamic_claude.py).
delta_T = 0.0       # 0 = use CFL-derived dt
T_final = 6000e-6     # total physical time (s)
v_imp   = 1.0      # imposed velocity (m/s)
T_ramp  = 1000e-6      # velocity ramp-up time (s)
rho_    = 8000.0    # density (kg/m^3)
nu_     = 0.3       # Poisson ratio
l_0     = 0.0       # 0 = auto (2*h_min)

g_c_ = 2.213e4
e_ = 190.0e9
domain_size = 1.0
# Job spec
num_nodes = 1
num_cores = 16
job_time = '12:00:00'

# log directory
# job script directory
os.makedirs(f'pfm_lite/jobs', exist_ok=True)

for sim_case_ in simulation_case:
    for model_ in ds_models:
        out_file = f'{sim_case_}_{model_}_{comment}'
        job_name = f'{sim_case_[0]}_{model_[0]}_{comment}'
        os.makedirs(f'pfm_lite/logs/{job_name}', exist_ok=True)
        job_script = f"""#!/bin/bash -l
#$ -pe mpi_{num_cores}_tasks_per_node {num_cores}
#$ -N {job_name}
#$ -l h_rt={job_time}
# Merge stderr into the stdout file, to reduce clutter.
#$ -j y
#$ -m beas
#$ -o pfm_lite/logs/{job_name}/
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
mpirun -np $NSLOTS python3 {python_file} --delta_T {delta_T} --T_final {T_final} --v_imp {v_imp} --T_ramp {T_ramp} --rho {rho_} --nu {nu_} --l_0 {l_0} --init_crack_add {init_crack_add} --mesh_size {mesh_size_} --prefix {out_file} --job_id $JOB_ID --seed $SGE_TASK_ID --g_c {g_c_} --e_ {e_} --domain_size {domain_size}
# Clean up task-specific cache
rm -rf $FFCX_CACHE_DIR
        """
        job_script_file = f'pfm_lite/jobs/{job_name}_{sim_case_}_{model_}_{mesh_size_}.sh'
        with open(job_script_file, 'w') as f:
            f.write(job_script)
        os.system(f'qsub {job_script_file}')

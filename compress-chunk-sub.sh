#!/bin/bash -l
#$ -N compress_job
#$ -l h_rt=4:00:00
#$ -j y
#$ -t 1-50:1          # 50 parallel tasks
#$ -o /projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/logs/compress/

module load miniconda
conda activate fenicsx-env
python3 compress_full_res.py --res 128 \
    --base_address "/projectnb/lejlab2/erfan/pfmv2/pfm_lite/pfm_dataset/results/tension_amor_3c" \
    --task_id $((SGE_TASK_ID - 1)) \
    --num_tasks 50
#!/usr/bin/env bash
#SBATCH --partition long
#SBATCH --account=mi2lab-hi
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time 0-00:30:00
#SBATCH --job-name=icm
#SBATCH --gres=gpu:2
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH --output=slurm_logs/icm-%A.log

# echo file content to logs
script_path=$(readlink -f "$0")
cat $script_path

# load slurm
source /etc/profile.d/slurm.sh

# allow worker connection
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# activate env
source /raid/shared/$USER/conda/etc/profile.d/conda.sh
conda activate icm

# run exp
cd /home2/faculty/bsobieski/icm

mpiexec -n 2 python test.py
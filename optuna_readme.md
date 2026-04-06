run_hpo_cluster.sh — SLURM cluster script
Two usage modes:

# Single job with explicit seed
sbatch run_hpo_cluster.sh --seed 42

# Job array — seed = SLURM_ARRAY_TASK_ID automatically
sbatch --array=42,123,456,789,1337 run_hpo_cluster.sh

# Local test
bash run_hpo_cluster.sh --seed 42 --no-wandb

Supports --seed, --n-trials, --wandb-group, --wandb-project, --no-wandb. Output goes to logs/hpo_{jobid}_{taskid}.out.
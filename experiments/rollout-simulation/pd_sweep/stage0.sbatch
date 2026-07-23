#!/bin/bash
# Submit the Stage 0 sweep as a four-hour batch job.
#   sbatch experiments/rollout-simulation/pd_sweep/stage0.sbatch
# Pass environment variables to select a subset:
#   ARMS="ep_off agrs" sbatch experiments/rollout-simulation/pd_sweep/stage0.sbatch
#
# Estimate: about 10 minutes to load plus four 8-minute concurrency runs per
# arm, or roughly three hours for four arms.
#SBATCH --job-name=pd-sweep-stage0
#SBATCH --account=coreai_horizon_dilations
#SBATCH --partition=batch
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/src/prime-rl-v0.7.0/outputs/rollout-simulation/sweep/stage0-slurm-%j.log

REPO=/scratch/fsw/portfolios/coreai/projects/coreai_horizon_dilations/users/kfujii/src/prime-rl-v0.7.0
# run_stage0.sh detects SLURM_JOB_ID and re-enters itself through Pyxis.
bash "$REPO/experiments/rollout-simulation/pd_sweep/run_stage0.sh"

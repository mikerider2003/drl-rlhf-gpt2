#!/bin/bash
#SBATCH -p GPU              # partition (queue)
#SBATCH -N 1                # number of nodes
#SBATCH -t 0-05:00          # time (D-HH:MM)
#SBATCH -o outputs/output.%N.%j.out  # STDOUT
#SBATCH -e outputs/output.%N.%j.err  # STDERR
#SBATCH --gres=gpu:1        # request 1 GPU

# Setup Conda environment
if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/usr/local/anaconda3/etc/profile.d/conda.sh"
else
    export PATH="/usr/local/anaconda3/bin:$PATH"
fi

# Activate your conda environment
conda activate drl_env

# Navigate to your project directory
cd ~/drl-rlhf-gpt2/part2_rlhf

# Run notebook, save outputs to a copy
jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  --output rlhf_sentiment_output.ipynb \
  rlhf_sentiment.ipynb
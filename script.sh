#!/bin/bash
#SBATCH --account=def-ssamet-ab
#SBATCH --gres=gpu:a100:1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=0-3:00
#SBATCH --mail-user=soham.ratnaparkhi@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Check if model name is provided
if [ $# -eq 0 ]; then
    $1=""
fi

# Set the output file name using the model name and job ID
#SBATCH --output=$1_%j.out

module load python
module load cuda

module load StdEnv/2023
module load cudacore/.12.2.2
module load arrow/17.0.0
module load gcc arrow/17.0.0

# Activate your virtual environment
source venv/bin/activate

# If you need to install any packages, uncomment and modify these lines:
# pip install --no-index torch torchvision
# pip install -e .
# pip install -r requirements.txt

# Run your Flower command
export PYTHONFAULTHANDLER=1
flwr run .
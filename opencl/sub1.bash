#!/bin/bash
#SBATCH -J MatrixMult
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o mul1.out
#SBATCH -e mul1.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sasol@oregonstate.edu
make benchfirst

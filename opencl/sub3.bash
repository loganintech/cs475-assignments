#!/bin/bash
#SBATCH -J MatrixMult
#SBATCH -A cs475-575
#SBATCH -p class
#SBATCH --gres=gpu:1
#SBATCH -o mul3.out
#SBATCH -e mul3.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=sasol@oregonstate.edu
make benchthird

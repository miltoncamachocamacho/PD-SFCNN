#!/bin/bash

###### Reserve computing resources ############# in gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=80G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

source ~/miniconda3/bin/activate myenv

python pd_sfcn.py \
#model name
model_name \
#is it the first run?
True \
#name of loaded model if any
model_loaded \
#n of epochs
60 \
#random split
1 \
#learning rate
0.01 \
#learning decay
0.001 \
#run augmentation
False \
#extension of DTI data to use for the run
ad,fa,md,rd \
#use age and sex?
True \
#add regularization
False \
#batch size
4 > log.txt

#for explainability also run
python exp.py \
#model name
model_name \
#is it the first run?
True \
#name of loaded model if any
model_loaded \
#n of epochs
60 \
#random split
1 \
#learning rate
0.01 \
#learning decay
0.001 \
#run augmentation
False \
#extension of DTI data to use for the run
ad,fa,md,rd \
#use age and sex?
True \
#add regularization
False \
#batch size
4 > log.txt

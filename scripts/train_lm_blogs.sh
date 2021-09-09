#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"

python src/lm_lstm.py \
    --dataset $1 \
    --style $2 \
    --automatic_multi_domain True \
    --max_decay $3 \
    #max dedcay used to be 12
    #--lr 0.000244 \
    #--resume_from pretrained_lm/blogs_3dom_style2/model.pt \
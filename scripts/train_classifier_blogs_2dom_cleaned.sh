#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

export PYTHONPATH="$(pwd)"

python src/cnn_classify.py \
  --dataset blogs_2dom_cleaned_distinguished \
  --output_dir "pretrained_classifer/blogs_2dom_cleaned_distinguished" \
  --clean_mem_every 5 \
  --reset_output_dir \
  --train_src_file data/blogs_2dom_cleaned_distinguished/train.txt \
  --train_trg_file data/blogs_2dom_cleaned_distinguished/train.attr \
  --dev_src_file data/blogs_2dom_cleaned_distinguished/dev.txt \
  --dev_trg_file data/blogs_2dom_cleaned_distinguished/dev.attr \
  --dev_trg_ref data/blogs_2dom_cleaned_distinguished/dev.txt \
  --src_vocab  data/blogs_2dom_cleaned_distinguished/text.vocab \
  --trg_vocab  data/blogs_2dom_cleaned_distinguished/attr_disc.vocab \
  --d_word_vec=128 \
  --d_model=512 \
  --log_every=100 \
  --eval_every=5000 \
  --out_c_list="1,2,3,4" \
  --k_list="3,3,3,3" \
  --batch_size 32 \
  --valid_batch_size=32 \
  --patience 5 \
  --lr_dec 0.8 \
  --dropout 0.3 \
  --cuda \
  --lr 0.001 \
  --classifer lstm \


#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

python src/main.py \
        --dataset multi_domain_renamed \
        --clean_mem_every 5 \
        --reset_output_dir \
        --classifier_dir="pretrained_classifer/multi_domain_renamed" \
        --train_src_file data/multi_domain_renamed/train.txt \
        --train_trg_file data/multi_domain_renamed/train.attr \
        --dev_src_file data/multi_domain_renamed/dev.txt \
        --dev_trg_file data/multi_domain_renamed/dev.attr \
        --dev_trg_ref data/multi_domain_renamed/dev_global.txt \
        --src_vocab  data/multi_domain_renamed/text.vocab \
        --trg_vocab  data/multi_domain_renamed/attr.vocab \
        --d_word_vec=128 \
        --d_model=512 \
        --log_every=100 \
        --eval_every=3000 \
        --ppl_thresh=10000 \
        --eval_bleu \
        --batch_size 32 \
        --valid_batch_size 128 \
        --patience 5 \
        --lr_dec 0.5 \
        --lr 0.001 \
        --dropout 0.3 \
        --max_len 10000 \
        --seed 0 \
        --beam_size 1 \
        --word_blank 0.2 \
        --word_dropout 0.1 \
        --word_shuffle 3 \
        --cuda \
        --anneal_epoch 5 \
        --temperature 0.01 \
        --klw 0.008 \
        --max_pool_k_size 1 \
        --bt \
        --lm \
        --gumbel_softmax \
        --avg_len \
        --no_styles 3 \
        --fl_len_control \
        --automated_multi_domain \
        #--one_lm \
        #--disced_lm \
        #--automated_multi_domain \
        #--one_lm \
        #--disced_lm \
        #--gs_soft \
        #--len_control \
        #--d_len_vec 1 \
        #--clip_grad 2 \
        #--automated_multi_domain \

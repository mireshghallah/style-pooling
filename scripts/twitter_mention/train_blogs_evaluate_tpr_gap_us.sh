#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=12g
##SBATCH --nodelist=compute-0-7
#SBATCH -t 0

python src/main.py \
        --dataset blogs_2dom \
        --clean_mem_every 5 \
        --reset_output_dir \
        --classifier_dir="pretrained_classifer/twitter_mention" \
        --train_src_file outputs_blogs_2dom_cleaned/blogs_2dom_cleaned_wd0.1_wb0.2_ws3.0_an5_pool1_klw0.04_lr0.0005_lrdec0.5_gs0.01_lm_bt_gs_hard_avglen_klrev2_flctrl_lenstop_boost1.0_revctlr_date_28_04_2021_21_04_54/dev.trans_15000 \
        --train_trg_file data/twitter_mention/test_doc_id.attr \
        --dev_src_file  outputs_blogs_2dom_cleaned/blogs_2dom_cleaned_wd0.1_wb0.2_ws3.0_an5_pool1_klw0.04_lr0.0005_lrdec0.5_gs0.01_lm_bt_gs_hard_avglen_klrev2_flctrl_lenstop_boost1.0_revctlr_date_28_04_2021_21_04_54/dev.trans_15000 \
        --dev_trg_file data/twitter_mention/test_doc_id.attr \
        --dev_trg_ref data/twitter_mention/test_text.txt \
        --src_vocab  data/blogs_2dom_cleaned/text.vocab \
        --trg_vocab  data/twitter_mention/test_doc_id.attr \
        --d_word_vec=128 \
        --d_model=512 \
        --log_every=100 \
        --eval_every=10000000 \
        --ppl_thresh=10000 \
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
        --klw 0.015 \
        --max_pool_k_size 1 \
        --bt \
        --lm \
        --gumbel_softmax \
        --avg_len \
        --no_styles 2 \
        --fl_len_control \
        --automated_multi_domain \
        --classifier_evaluation_tpr_gap \
        --input_doc_dict "data/twitter_mention/test_mention_id.attr" \
        --input_doc_dict_sens "data/twitter_mention/test_dom.attr" \
        #--element_wise_all_kl \
        #--eval_bleu \
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

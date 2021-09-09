export PYTHONPATH="$(pwd)"

python src/lm_lstm.py \
    --dataset multi_domain_renamed \
    --eval_from pretrained_lm/multi_domain_renamed_style$1/model.pt \
    --test_trg_file data/multi_domain_renamed/dev.attr \
    --test_src_file outputs_multi_domain_renamed/multi_domain_renamed_wd0.1_wb0.2_ws3.0_an5_pool1_klw0.02_lr0.001_lrdec0.5_gs0.01_lm_bt_gs_hard_avglen_klrev3_flctrl_date_05_02_2021_15_34_59/dev.trans_6000 \
    --automatic_multi_domain True \
    --style $1 \
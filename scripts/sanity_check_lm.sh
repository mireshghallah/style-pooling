export PYTHONPATH="$(pwd)"

python src/lm_lstm.py \
    --dataset blogs_2dom_distinguished \
    --eval_from pretrained_lm/blogs_2dom_distinguished_style2/model.pt \
    --test_trg_file data/blogs_2dom_distinguished/dev_select_age_1.attr \
    --test_src_file data/blogs_2dom_distinguished/blogs.dev_cleaned_select.age.1 \
    --automatic_multi_domain True \
    --style 2 \
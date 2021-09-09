export PYTHONPATH="$(pwd)"

python src/lm_lstm.py \
    --dataset blogs_2dom \
    --eval_from pretrained_lm/blogs_2dom_style1/model.pt \
    --test_trg_file data/blogs_2dom_distinguished/dev_select_age_0.attr \
    --test_src_file data/blogs_2dom_distinguished/blogs.dev_cleaned_select.age.0 \
    --automatic_multi_domain True \
    --style 1 \
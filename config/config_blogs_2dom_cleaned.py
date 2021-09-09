params=[{
  "d_word_vec": 1024,
  "d_model": 2048,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/blogs_2dom_cleaned/blogs.train_cleaned.age.0",
  "train_trg_file": "data/blogs_2dom_cleaned/train_age_0.attr",
  "dev_src_file": "data/blogs_2dom_cleaned/blogs.dev_cleaned.age.0",
  "dev_trg_file": "data/blogs_2dom_cleaned/dev_age_0.attr",
  "src_vocab": "data/blogs_2dom_cleaned/text.vocab",
  "trg_vocab": "data/blogs_2dom_cleaned/attr.vocab"
},
{
  "d_word_vec": 1024,
  "d_model": 2048,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/blogs_2dom_cleaned/blogs.train_cleaned.age.1",
  "train_trg_file": "data/blogs_2dom_cleaned/train_age_1.attr",
  "dev_src_file": "data/blogs_2dom_cleaned/blogs.dev_cleaned.age.1",
  "dev_trg_file": "data/blogs_2dom_cleaned/dev_age_1.attr",
  "src_vocab": "data/blogs_2dom_cleaned/text.vocab",
  "trg_vocab": "data/blogs_2dom_cleaned/attr.vocab"
}
,#for wrong (reverse, indist) lm
{
  "d_word_vec": 1024,
  "d_model": 2048,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/blogs_2dom_cleaned/train_select.txt",
  "train_trg_file": "data/blogs_2dom_cleaned/train_select.attr",
  "dev_src_file": "data/blogs_2dom_cleaned/dev_select.txt",
  "dev_trg_file": "data/blogs_2dom_cleaned/dev_select.attr",
  "src_vocab": "data/blogs_2dom_cleaned/text.vocab",
  "trg_vocab": "data/blogs_2dom_cleaned/attr.vocab"
}, #yelp data
{
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/multi_domain_renamed/train_global.txt",
  "train_trg_file": "data/multi_domain_renamed/train.attr",
  "dev_src_file": "data/blogs_2dom/dev_drop_20.txt",
  "dev_trg_file": "data/blogs_2dom/dev_drop_20.attr",
  "src_vocab": "data/blogs_2dom_cleaned/text.vocab",
  "trg_vocab": "data/multi_domain_renamed/attr.vocab"
},
 #for discriminator
 {
 "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/blogs_2dom_cleaned/train.txt",
  "train_trg_file": "data/blogs_2dom_cleaned/train.attr",
  "dev_src_file": "data/blogs_2dom_cleaned/dev_drop_20.txt",
  "dev_trg_file": "data/blogs_2dom_cleaned/dev_drop_20.attr",
  "src_vocab": "data/blogs_2dom_cleaned/text.vocab",
  "trg_vocab": "data/blogs_2dom_cleaned/attr_disc.vocab"
 },
#global model
  {
 "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/blogs_2dom_cleaned/train_global.txt",
  "train_trg_file": "data/blogs_2dom_cleaned/train.attr",
  "dev_src_file": "data/blogs_2dom_cleaned/dev_global.txt",
  "dev_trg_file": "data/blogs_2dom_cleaned/dev.attr",
  "src_vocab": "data/blogs_2dom_cleaned/text.vocab",
  "trg_vocab": "data/blogs_2dom_cleaned/attr.vocab"
 }
]

params_disc={
  "arch":"other",
  "hidden_d":512,
  "out_c_list":"1,2,3,4",
  "k_list":"1,1,1,1",
  "trg_vocab_size":3,
  "lr":0.05,
  "lam":10,
  "cuda":True,
  "init_range":1.0,
  "hidden_size":1024,
  "hidden_size2":256
}

params_main={
  "lm_style":["pretrained_lm/blogs_2dom_cleaned_style0/model.pt", "pretrained_lm/blogs_2dom_cleaned_style1/model.pt", "pretrained_lm/blogs_2dom_cleaned_style2/model.pt"], #change the last one to 2 for indist and 3 for yelp
  "eval_cls": True
}

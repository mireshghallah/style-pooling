params=[{
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/multi_domain_renamed/multi_domain_renamed.train.0",
  "train_trg_file": "data/multi_domain_renamed/train_0.attr",
  "dev_src_file": "data/multi_domain_renamed/multi_domain_renamed.dev.0",
  "dev_trg_file": "data/multi_domain_renamed/dev_0.attr",
  "src_vocab": "data/multi_domain_renamed/text.vocab",
  "trg_vocab": "data/multi_domain_renamed/attr.vocab"
},
{
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/multi_domain_renamed/multi_domain_renamed.train.1",
  "train_trg_file": "data/multi_domain_renamed/train_1.attr",
  "dev_src_file": "data/multi_domain_renamed/multi_domain_renamed.dev.1",
  "dev_trg_file": "data/multi_domain_renamed/dev_1.attr",
  "src_vocab": "data/multi_domain_renamed/text.vocab",
  "trg_vocab": "data/multi_domain_renamed/attr.vocab"
}
,
{
  "d_word_vec": 128,
  "d_model": 512,
  "log_every": 100,
  "eval_every": 2500,
  "batch_size": 32,
  "dropout_in": 0.3,
  "dropout_out": 0.3,
  "train_src_file": "data/multi_domain_renamed/multi_domain_renamed.train.2",
  "train_trg_file": "data/multi_domain_renamed/train_2.attr",
  "dev_src_file": "data/multi_domain_renamed/multi_domain_renamed.dev.2",
  "dev_trg_file": "data/multi_domain_renamed/dev_2.attr",
  "src_vocab": "data/multi_domain_renamed/text.vocab",
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
  "train_src_file": "data/multi_domain_renamed/train.txt",
  "train_trg_file": "data/multi_domain_renamed/train.attr",
  "dev_src_file": "data/multi_domain_renamed/dev.txt",
  "dev_trg_file": "data/multi_domain_renamed/dev.attr",
  "src_vocab": "data/multi_domain_renamed/text.vocab",
  "trg_vocab": "data/multi_domain_renamed/attr.vocab"
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
  "train_src_file": "data/multi_domain_renamed/train_global.txt",
  "train_trg_file": "data/multi_domain_renamed/train.attr",
  "dev_src_file": "data/multi_domain_renamed/dev_global.txt",
  "dev_trg_file": "data/multi_domain_renamed/dev.attr",
  "src_vocab": "data/multi_domain_renamed/text.vocab",
  "trg_vocab": "data/multi_domain_renamed/attr.vocab"
 }
]

params_disc={
  "arch":"other",
  "hidden_d":512,
  "out_c_list":"1,2,3,4",
  "k_list":"1,1,1,1",
  "trg_vocab_size":3,
  "lr":0.05,
  "lam":0.1,
  "cuda":True,
  "init_range":1.0,
  "hidden_size":1024,
  "hidden_size2":256
}

params_main={
  "lm_style":["pretrained_lm/multi_domain_renamed_style0/model.pt", "pretrained_lm/multi_domain_renamed_style1/model.pt", "pretrained_lm/multi_domain_renamed_style2/model.pt","pretrained_lm/multi_domain_renamed_style3/model.pt"],
  "eval_cls": True
}

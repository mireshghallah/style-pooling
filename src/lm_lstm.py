import sys
import os
import time
import argparse
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import cnn_classify

from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from data_utils import DataUtil
from hparams import *
from utils import *


import numpy as np

clip_grad = 5.0
decay_step = 5
lr_decay = 0.5
#max_decay = 5

if __name__ == "__main__":
    from sys import path
    from os.path import dirname as dir

    path.append(dir(path[0]))

class LSTM_LM(nn.Module):
  """LSTM language model"""
  def __init__(self, model_init, emb_init, hparams):
    super(LSTM_LM, self).__init__()
    self.nh = hparams.d_model
    self.hparams = hparams
    # no padding when setting padding_idx to -1
    self.embed = nn.Embedding(hparams.src_vocab_size, 
      hparams.d_word_vec, padding_idx=hparams.pad_id)

    self.dropout_in = nn.Dropout(hparams.dropout_in)
    self.dropout_out = nn.Dropout(hparams.dropout_out)

    # concatenate z with input
    self.lstm = nn.LSTM(input_size=hparams.d_word_vec,
                 hidden_size=hparams.d_model,
                 num_layers=1,
                 batch_first=True)

    # prediction layer
    self.pred_linear = nn.Linear(self.nh, hparams.src_vocab_size, bias=True)

    if hparams.tie_weight:
        self.pred_linear.weight = self.embed.weight

    self.loss = nn.CrossEntropyLoss(ignore_index=hparams.pad_id, reduction="none")

    self.reset_parameters(model_init, emb_init)

  def reset_parameters(self, model_init, emb_init):
    for param in self.parameters():
      model_init(param)
    emb_init(self.embed.weight)

    self.pred_linear.bias.data.zero_()


  def decode(self, x, x_len, gumbel_softmax=False):
    """
    Args:
      x: (batch_size, seq_len)
      x_len: list of x lengths
    """

    # not predicting start symbol
    # sents_len -= 1

    if gumbel_softmax:
      batch_size, seq_len, _ = x.size()
      word_embed = x @ self.embed.weight
    else:
      batch_size, seq_len = x.size()

      # (batch_size, seq_len, ni)
      word_embed = self.embed(x)

    word_embed = self.dropout_in(word_embed)
    packed_embed = pack_padded_sequence(word_embed, x_len, batch_first=True)
    
    c_init = word_embed.new_zeros((1, batch_size, self.nh))
    h_init = word_embed.new_zeros((1, batch_size, self.nh))
    output, __ = self.lstm(packed_embed, (h_init, c_init))
    output, _ = pad_packed_sequence(output, batch_first=True)

    output = self.dropout_out(output)

    # (batch_size, seq_len, vocab_size)
    output_logits = self.pred_linear(output)
    if hasattr(self, "hparams") and self.hparams.use_discriminator:
      return output_logits, __
    else:
      return output_logits

  def reconstruct_error(self, x, x_len, gumbel_softmax=False, x_mask=None):
    """Cross Entropy in the language case
    Args:
      x: (batch_size, seq_len)
      x_len: list of x lengths
      x_mask: required if gumbel_softmax is True, 1 denotes mask,
              size (batch_size, seq_len)
    Returns:
      loss: (batch_size). Loss across different sentences
    """

    #remove end symbol
    src = x[:, :-1]

    # remove start symbol
    tgt = x[:, 1:]

    if gumbel_softmax:
      batch_size, seq_len, _ = src.size()
    else:
      batch_size, seq_len = src.size()

    x_len = [s - 1 for s in x_len]

    # (batch_size, seq_len, vocab_size)
    output_logits = self.decode(src, x_len, gumbel_softmax)

    if gumbel_softmax:
      log_p = F.log_softmax(output_logits, dim=2)
      x_mask = x_mask[:, 1:]
      loss = -((log_p * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1)
    else:
      tgt = tgt.contiguous().view(-1)
      # (batch_size * seq_len)
      loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                 tgt)
      loss = loss.view(batch_size, -1).sum(-1)


    # (batch_size)
    return loss

  def reconstruct_error_disc(self, x, x_len, y, disc, eval=False, gumbel_softmax=False, x_mask=None):
    """Cross Entropy in the language case
    Args:
      x: (batch_size, seq_len)
      x_len: list of x lengths
      x_mask: required if gumbel_softmax is True, 1 denotes mask,
              size (batch_size, seq_len)
    Returns:
      loss: (batch_size). Loss across different sentences
    """

    #remove end symbol
    src = x[:, :-1]

    # remove start symbol
    tgt = x[:, 1:]

    if gumbel_softmax:
      batch_size, seq_len, _ = src.size()
    else:
      batch_size, seq_len = src.size()

    x_len = [s - 1 for s in x_len]

    # (batch_size, seq_len, vocab_size)
    output_logits, hidden_state = self.decode(src, x_len, gumbel_softmax)

    #neut disc for LM
    disc.eval()
    if disc.hparams.arch == "simplenet":
      disc_out = disc(hidden_state[0].squeeze(0))
    else:
      disc_out = disc(hidden_state[0].permute(1,2,0))
    #neut_loss =(-1/(disc_out.shape[0]*disc_out.shape[1])*(torch.sum(torch.log(torch.nn.functional.softmax(disc_out)))))
    neut_loss = -1/(disc_out.shape[0]*disc_out.shape[1])*(torch.sum(torch.log(1e-28+ torch.nn.functional.relu(torch.nn.functional.softmax(disc_out) - 1e-28 ))))
    #train disc for disc
    crit = nn.CrossEntropyLoss()
    if not eval:
      disc.train()
    if disc.hparams.arch == "simplenet":
      disc_train_out = disc(hidden_state[0].detach().squeeze(0))
    else :
      disc_train_out = disc(hidden_state[0].detach().permute(1,2,0))
    #print(disc_train_out.shape, torch.squeeze(y).shape)
    disc_loss = crit(disc_train_out, torch.squeeze(y))

    #accuracy of disc
    _, predicted = torch.max(disc_train_out.detach().data, 1)
    batch_correct = (predicted == torch.squeeze(y)).sum().item()

    if gumbel_softmax:
      log_p = F.log_softmax(output_logits, dim=2)
      x_mask = x_mask[:, 1:]
      loss = -((log_p * tgt).sum(dim=2) * (1. - x_mask)).sum(dim=1)
    else:
      tgt = tgt.contiguous().view(-1)
      # (batch_size * seq_len)
      loss = self.loss(output_logits.view(-1, output_logits.size(2)),
                 tgt)
      loss = loss.view(batch_size, -1).sum(-1)


    # (batch_size)
    return loss, neut_loss, disc_loss, batch_correct


  def compute_gumbel_logits(self, x, x_len):
    """Cross Entropy in the language case
    Args:
      x: (batch_size, seq_len)
      x_len: list of x lengths
      x_mask: required if gumbel_softmax is True, 1 denotes mask,
              size (batch_size, seq_len)
    Returns:
      loss: (batch_size). Loss across different sentences
    """

    #remove end symbol
    src = x[:, :-1]

    batch_size, seq_len, _ = src.size()

    x_len = [s - 1 for s in x_len]

    # (batch_size, seq_len, vocab_size)
    output_logits = self.decode(src, x_len, True)

    # (batch_size)
    return output_logits

  def log_probability(self, x, x_len, gumbel_softmax=False, x_mask=None):
    """Cross Entropy in the language case
    Args:
      x: (batch_size, seq_len)
    Returns:
      log_p: (batch_size).
    """

    return -self.reconstruct_error(x, x_len, gumbel_softmax, x_mask)

def init_args():
  parser = argparse.ArgumentParser(description='language model')
  parser.add_argument('--seed', type=int, default=783435, metavar='S', help='random seed')
  parser.add_argument('--max_decay', type=int, default=5, help='number of times to decay lr')
  parser.add_argument('--lr', type=float, default=1.0, help='start lr')
  parser.add_argument('--dataset', type=str, help='dataset name')
  parser.add_argument('--eval_from', type=str, default="", help="eval pre-trained model")
  parser.add_argument('--resume_from', type=str, default="", help="resume half-trained model")
  parser.add_argument('--style', type=int, help='binary, 0 or 1')
  parser.add_argument("--decode", action="store_true", help="whether to decode only")
  parser.add_argument("--max_len", type=int, default=10000, help="maximum len considered on the target side")
  parser.add_argument("--tie_weight", action="store_true", help="whether use embedding weight in the pre-softmax")
  parser.add_argument("--output", type=str, default="")

  parser.add_argument('--test_src_file', type=str, default="")
  parser.add_argument('--test_trg_file', type=str, default="")
  parser.add_argument("--shuffle_train", action="store_true", help="load an existing model")
  parser.add_argument("--automatic_multi_domain", type=bool, default=False) #for backward compatibility
  parser.add_argument("--use_discriminator", type=bool, default=False) #whether to use discriminator


  args = parser.parse_args()
  args.cuda = torch.cuda.is_available()

  if args.eval_from == "":
    args.output_dir = "pretrained_lm/{}_style{}/".format(args.dataset, args.style)
  else:
    args.output_dir = "pretrained_lm/{}_eval_style{}/".format(args.dataset, args.style)

  if args.output != "":
    args.output_dir = args.output

  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

  args.device = torch.device("cuda" if args.cuda else "cpu")

  config_file = "config.config_{}".format(args.dataset)

  if (not args.automatic_multi_domain):
    if args.style == 0:
      params = importlib.import_module(config_file).params0
    else:
      params = importlib.import_module(config_file).params1
  else:
    params = importlib.import_module(config_file).params[args.style]


  if args.use_discriminator:
    params_disc = importlib.import_module(config_file).params_disc
    args_disc = argparse.Namespace(**params_disc)

  args = argparse.Namespace(**params, **vars(args))

  if args.eval_from != "":
    args.dev_src_file = args.test_src_file
    args.dev_trg_file = args.test_trg_file

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.cuda:
      torch.cuda.manual_seed(args.seed)
      torch.backends.cudnn.deterministic = True
  if args.use_discriminator:
    return args, args_disc
  else:
    return args, None

def test(model, data, hparams):
  model.eval()
  report_words = report_loss = report_ppl = report_sents = 0
  
  ppl_file_name = hparams.test_trg_file+"_lm_{}_score".format(str(hparams.style))
  ppl_file = open(ppl_file_name, 'w+')

  
  while True:
    x_valid, x_mask, x_count, x_len, \
    x_pos_emb_idxs, y_valid, y_mask, \
    y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_dev(dev_batch_size=hparams.batch_size)

    report_words += (x_count - batch_size)
    report_sents += batch_size

    loss = model.reconstruct_error(x_valid, x_len)

    report_loss += loss.sum().item()

    del x_valid
    del x_mask
    del x_count
    del x_len
    del x_pos_emb_idxs
    del loss
    torch.cuda.empty_cache()

    if end_of_epoch:
      break

  log_string = "\n-----------VAL-----------\n"

  log_string += " VAL loss={0:<7.2f}".format(report_loss / report_sents)
  log_string += " VAL ppl={0:<8.2f}".format(np.exp(report_loss / report_words))

  log_string += "\n-----------VAL-----------\n"

  print(log_string)

  return report_loss / report_sents, np.exp(report_loss / report_words)

def train(args):

  class uniform_initializer(object):
      def __init__(self, stdv):
          self.stdv = stdv
      def __call__(self, tensor):
          nn.init.uniform_(tensor, -self.stdv, self.stdv)

  opt_dict = {"not_improved": 0, "lr": args.lr, "best_ppl": 1e4}

  hparams = HParams(**vars(args))
  data = DataUtil(hparams=hparams)

  model_init = uniform_initializer(0.01)
  emb_init = uniform_initializer(0.1)

  model = LSTM_LM(model_init, emb_init, hparams)

  if hparams.eval_from != "":
    model = torch.load(hparams.eval_from)
    model.to(hparams.device)
    with torch.no_grad():
        test(model, data, hparams)

    return 

  if hparams.resume_from !="":
    model = torch.load(hparams.resume_from)

  model.to(hparams.device)

  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  optim = torch.optim.SGD(model.parameters(), lr=hparams.lr)

  step = epoch = decay_cnt = 0
  report_words = report_loss = report_ppl = report_sents = 0
  start_time = time.time()

  model.train()
  while True:
    x_train, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y_train, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_sampled, y_sampled_mask, y_sampled_count, y_sampled_len, \
    y_pos_emb_idxs, batch_size,  eop = data.next_train()


    report_words += (x_count - batch_size)
    report_sents += batch_size

    optim.zero_grad()

    loss = model.reconstruct_error(x_train, x_len)


    loss = loss.mean(dim=-1)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    optim.step()

    report_loss += loss.item() * batch_size

    del x_train
    del x_mask
    del x_count
    del x_len
    del x_pos_emb_idxs
    del loss
    torch.cuda.empty_cache()


    if step % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0

      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={}".format(step)
      log_string += " lr={0:<9.7f}".format(opt_dict["lr"])
      log_string += " loss={0:<7.2f}".format(report_loss / report_sents)
      log_string += " |g|={0:<5.2f}".format(grad_norm)

      log_string += " ppl={0:<8.2f}".format(np.exp(report_loss / report_words))

      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)

    if step % args.eval_every == 0:
      with torch.no_grad():
        val_loss, val_ppl = test(model, data, hparams)
        if val_ppl < opt_dict["best_ppl"]:
          print("update best ppl")
          opt_dict["best_ppl"] = val_ppl
          opt_dict["not_improved"] = 0
          torch.save(model, os.path.join(hparams.output_dir, "model.pt"))

        if val_ppl > opt_dict["best_ppl"]:
          opt_dict["not_improved"] += 1
          if opt_dict["not_improved"] >= decay_step:
            opt_dict["not_improved"] = 0
            opt_dict["lr"] = opt_dict["lr"] * lr_decay
            model = torch.load(os.path.join(hparams.output_dir, "model.pt"))
            print("new lr: {0:<9.7f}".format(opt_dict["lr"]))
            decay_cnt += 1
            optim = torch.optim.SGD(model.parameters(), lr=opt_dict["lr"])

      report_words = report_loss = report_ppl = report_sents = 0
      model.train()

    step += 1
    if eop:
      epoch += 1

    if decay_cnt >= hparams.max_decay:
      break


def test_discriminator(model, disc, data, hparams):
  model.eval()
  report_words = report_loss = report_ppl = report_sents = 0
  corr_disc = disc_avg_loss = neut_avg_loss = all_disc_samples = 0

  while True:
    x_valid, x_mask, x_count, x_len, \
    x_pos_emb_idxs, y_valid, y_mask, \
    y_count, y_len, y_pos_emb_idxs, \
    y_neg, batch_size, end_of_epoch, _ = data.next_dev(dev_batch_size=hparams.batch_size)

    report_words += (x_count - batch_size)
    report_sents += batch_size
    all_disc_samples += batch_size

    loss, _, disc_loss, correct = model.reconstruct_error_disc(x_valid, x_len, y_valid, disc, eval=True)

    report_loss += loss.sum().item()
    
    disc_avg_loss += disc_loss.item() * batch_size
    corr_disc += correct

    if end_of_epoch:
      break

  log_string = "\n-----------VAL-----------\n"

  log_string += " VAL loss={0:<7.2f}".format(report_loss / report_sents)
  log_string += " VAL ppl={0:<8.2f}".format(np.exp(report_loss / report_words))

  log_string += " discriminator loss={0:<7.2f}".format(disc_avg_loss / all_disc_samples)
  log_string += " neut loss={0:<7.2f}".format(neut_avg_loss / all_disc_samples)
  log_string += " disc accuracy={0:<7.2f}".format(corr_disc / all_disc_samples*100)


  log_string += "\n-----------VAL-----------\n"

  print(log_string)

  return report_loss / report_sents, np.exp(report_loss / report_words), disc_avg_loss/all_disc_samples, corr_disc/all_disc_samples*100



def train_discriminator(args, args_disc):

  class uniform_initializer(object):
      def __init__(self, stdv):
          self.stdv = stdv
      def __call__(self, tensor):
          nn.init.uniform_(tensor, -self.stdv, self.stdv)

  opt_dict = {"not_improved": 0, "lr": args.lr, "best_ppl": 1e4}

  hparams = HParams(**vars(args))
  hparams_disc = HParams(**vars(args_disc))

  data = DataUtil(hparams=hparams)

  model_init = uniform_initializer(0.01)
  emb_init = uniform_initializer(0.1)

  model = LSTM_LM(model_init, emb_init, hparams)

  ###setup disc
  if hparams_disc.arch == "simplenet":
    disc =cnn_classify.simplenet(hparams_disc)
    print("Disc arch is simplenet")
  else:
    disc = cnn_classify.CNNClassifyWOEmb(hparams_disc)
  

  
  if hparams.eval_from != "":
    model = torch.load(hparams.eval_from)
    model.to(hparams.device)
    with torch.no_grad():
        test(model, data, hparams)

    return 

  model.to(hparams.device)
  disc.to(hparams.device)


  trainable_params = [
    p for p in model.parameters() if p.requires_grad]
  num_params = count_params(trainable_params)
  print("Model has {0} params".format(num_params))

  optim = torch.optim.SGD(model.parameters(), lr=hparams.lr)

  ####discriminator
  trainable_params_disc = [
      p for p in disc.parameters() if p.requires_grad]
  num_params_disc = count_params(trainable_params_disc)
  print("Discriminator has {0} params".format(num_params_disc))

  optim_disc = torch.optim.Adam(trainable_params_disc, lr=hparams_disc.lr)
  ## 


  step = epoch = decay_cnt = 0
  
  report_words = report_loss = report_ppl = report_sents = 0
  corr_disc = disc_avg_loss = neut_avg_loss = all_disc_samples = 0

  start_time = time.time()

  model.train()



  while True:
    x_train, x_mask, x_count, x_len, x_pos_emb_idxs, \
    y_train, y_mask, y_count, y_len, y_pos_emb_idxs, \
    y_sampled, y_sampled_mask, y_sampled_count, y_sampled_len, \
    y_pos_emb_idxs, batch_size,  eop = data.next_train()


    report_words += (x_count - batch_size)
    report_sents += batch_size
    all_disc_samples += batch_size

    optim.zero_grad()
    optim_disc.zero_grad()

    disc.eval()
    loss, neut_loss, disc_loss, correct = model.reconstruct_error_disc(x_train, x_len, y_train, disc)

    loss = loss.mean(dim=-1)
    report_loss += loss.item() * batch_size #TODO check this
    loss += disc.hparams.lam * neut_loss
    
    loss.backward()
    disc_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    optim.step()
    optim_disc.step()

    
    disc_avg_loss += disc_loss.item() * batch_size
    neut_avg_loss += neut_loss.item() * batch_size
    corr_disc += correct

    if step % args.log_every == 0:
      curr_time = time.time()
      since_start = (curr_time - start_time) / 60.0

      log_string = "ep={0:<3d}".format(epoch)
      log_string += " steps={}".format(step)
      log_string += " lr={0:<9.7f}".format(opt_dict["lr"])
      log_string += " loss={0:<7.2f}".format(report_loss / report_sents)
      
      log_string += " discriminator loss={0:<7.2f}".format(disc_avg_loss / all_disc_samples)
      log_string += " neut loss={0:<7.2f}".format(neut_avg_loss / all_disc_samples)
      log_string += " disc accuracy={0:<7.2f}".format(corr_disc / all_disc_samples*100)


      log_string += " |g|={0:<5.2f}".format(grad_norm)

      log_string += " ppl={0:<8.2f}".format(np.exp(report_loss / report_words))

      log_string += " time(min)={0:<5.2f}".format(since_start)
      print(log_string)

    if step % args.eval_every == 0:
      with torch.no_grad():
        val_loss, val_ppl, loss_disc, acc_disc = test_discriminator(model, disc, data, hparams)
        if val_ppl < opt_dict["best_ppl"]:
          print("update best ppl")
          opt_dict["best_ppl"] = val_ppl
          opt_dict["not_improved"] = 0
          torch.save(model, os.path.join(hparams.output_dir, "model.pt"))

        if val_ppl > opt_dict["best_ppl"]:
          opt_dict["not_improved"] += 1
          if opt_dict["not_improved"] >= decay_step:
            opt_dict["not_improved"] = 0
            opt_dict["lr"] = opt_dict["lr"] * lr_decay
            model = torch.load(os.path.join(hparams.output_dir, "model.pt"))
            print("new lr: {0:<9.7f}".format(opt_dict["lr"]))
            decay_cnt += 1
            optim = torch.optim.SGD(model.parameters(), lr=opt_dict["lr"])

      report_words = report_loss = report_ppl = report_sents = 0
      model.train()

    step += 1
    if eop:
      epoch += 1

    if decay_cnt >= hparams.max_decay:
      break




if __name__ == '__main__':
  args, args_disc = init_args()
  log_file = os.path.join(args.output_dir, "stdout")
  print("Logging to {}".format(log_file))
  sys.stdout = Logger(log_file)
  print(args)
  if args.use_discriminator:
    train_discriminator(args, args_disc)
  else:
    train(args)

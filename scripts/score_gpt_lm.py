import torch
import sys
import numpy as np
import os
from string import punctuation
 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Load pre-trained model (weights)

 
def score(sentence, model, tokenizer):
    


    tokenize_input = tokenizer.encode(sentence)
    if len(tokenize_input) == 0:
        return np.nan
    tensor_input = torch.tensor([tokenize_input])
    loss=model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())

def score_many_sents(directory, step):
    with torch.no_grad():
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
# Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    input_address =  os.path.join(directory, "dev.trans_{0}".format(step))
    output_score =  os.path.join(directory, "dev.trans_{0}_score".format(step))
    list_scores = []
    with open(input_address, 'r') as in_file, open(output_score, 'w+') as out_file:
        for i, line in enumerate(in_file):
            score_sent = score(line[:-1], model, tokenizer)
            list_scores.append(score_sent)
            
            out_file.write(str(score_sent))
            out_file.write('\n')
            #print(line[:-1])
            #print (score_sent)
            
            if (i%10000 == 0):
                print(i)
                #exit(0)
    print(float(sum(list_scores))/float(len(list_scores)))
    avg_score = (float(sum(list_scores))/float(len(list_scores)))
    return list_scores, avg_score


def score_many_sents_strip(directory, step):
    with torch.no_grad():
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
# Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    input_address =  os.path.join(directory, "dev.trans_{0}".format(step))
    output_score =  os.path.join(directory, "dev.trans_{0}_score_strip".format(step))
    list_scores = []
    with open(input_address, 'r') as in_file, open(output_score, 'w+') as out_file:
        for i, line in enumerate(in_file):
            
            score_sent = score(line[:-1].strip(punctuation), model, tokenizer)
            list_scores.append(score_sent)
            
            out_file.write(str(score_sent))
            out_file.write('\n')
            #print(line[:-1])
            #print (score_sent)
            
            if (i%10000 == 0):
                print(i)
                #exit(0)
    print(float(sum(list_scores))/float(len(list_scores)))
    avg_score = (float(sum(list_scores))/float(len(list_scores)))
    return list_scores, avg_score

#score_many_sents("data/doc_blogs_2dom",0)


def score_many_sents_dot(directory, step):
    with torch.no_grad():
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
# Load pre-trained model tokenizer (vocabulary)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    input_address =  os.path.join(directory, "dev.trans_{0}".format(step))
    output_score =  os.path.join(directory, "dev.trans_{0}_score_strip".format(step))
    list_scores = []
    with open(input_address, 'r') as in_file, open(output_score, 'w+') as out_file:
        for i, line in enumerate(in_file):
            
            score_sent = score(line[:-1].strip(punctuation)+".", model, tokenizer)
            list_scores.append(score_sent)
            
            out_file.write(str(score_sent))
            out_file.write('\n')
            #print(line[:-1])
            #print (score_sent)
            
            if (i%10000 == 0):
                print(i)
                #exit(0)
    print(float(sum(list_scores))/float(len(list_scores)))
    avg_score = (float(sum(list_scores))/float(len(list_scores)))
    return list_scores, avg_score
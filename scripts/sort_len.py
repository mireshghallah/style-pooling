import argparse
import nltk.data
import os
import json
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
#from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
import re
import unicodedata



def sort_len(args):

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


    input_file = open(args.input, 'r')
    input_attr_file = open(args.input_attrs, 'r')

    output_file = open(args.output, 'w+')
    output_order_file = open(args.output_order, 'w+')


    #AGE

    #save the train set in related files
    for line in input_file:
        sent = line.split()#tokenizer.tokenize(line[:-1])
        print(sent)
        exit(0)
        age =   get_age_group(int(instance['age']))
        if age is None:
            continue
        age_num = (instance['age'])
        gender = GENDER_DICT[instance['gender']]
        for sent in sents:
            processed_sents = preprocess(sent, args)
            for processed_sent in processed_sents:
                if len(word_tokenize(processed_sent))>args.min_token:
                #write to all file
                    all_train.write(processed_sent)
                    all_train.write('\n')
                    #write to trg all file
                    all_train_trg_age.write(age_num)
                    all_train_trg_age.write('\n')
                    all_train_trg_gender.write(str(gender))
                    all_train_trg_gender.write('\n')
                    #write to corrrosponding train file
                    train_age_files[age].write(processed_sent)
                    train_age_files[age].write('\n')
                    train_gender_files[gender].write(processed_sent)
                    train_gender_files[gender].write('\n')
    print("done writing training files")
    #save the dev set in related files
    for instance in test_set:
        sents = tokenizer.tokenize(instance['post'])
        age =   get_age_group(int(instance['age']))
        if age is None:
            continue
        age_num = (instance['age'])
        gender = GENDER_DICT[instance['gender']]
        
        for sent in sents:
            processed_sents = preprocess(sent, args)
            for processed_sent in processed_sents:
                if len(word_tokenize(processed_sent))>args.min_token:
                #write to all file
                    all_dev.write(processed_sent)
                    all_dev.write('\n')
                    #write to trg all file
                    all_dev_trg_age.write(age_num)
                    all_dev_trg_age.write('\n')
                    all_dev_trg_gender.write(str(gender))
                    all_dev_trg_gender.write('\n')
                    #write to corrrosponding train file
                    dev_age_files[age].write(processed_sent)
                    dev_age_files[age].write('\n')
                    dev_gender_files[gender].write(processed_sent)
                    dev_gender_files[gender].write('\n')

    print("done writing dev files")

    



    



def main(flags=None):
    """Main method for `id_extractor.py`

    Args:
        flags (List[str], optional): command line flags, useful for debugging. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    

    parser.add_argument("--input", type=str, help="input directory", default="")
    parser.add_argument("--input_attrs", type=str, help="input directory", default="")


    parser.add_argument("--output", type=str, help="output directory", default="")
    parser.add_argument("--output_order", type=str, help="output order directory", default="")
    parser.add_argument("--output_attrs", type=str, help="input directory", default="")




    args = parser.parse_args(flags)
    
    sort_len(args)
    #log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "successfully extracted raw ids")

if __name__ == '__main__':
    main()
    # example usage: python -m smartcompose_dp.neural_lang_model.lib.preprocessing.id_extractor --input /home/t-famire/json_avocado.json --clean_data /home/t-famire/avocado.dat  --output_encoded_ids /home/t-famire/encoded_ids.txt  --raw_ids=True --output_raw_ids=/home/t-famire/ids.txt


import argparse
import os

def metrics(args):

    vocab = []
    with open(args.vocab, 'r', encoding='utf-8') as vocab_fh:
        for token in vocab_fh:
            vocab.append(token.strip())
    vocab_size = len(vocab)
    #print("vocab_size is: ", vocab_size)

    ood_words = [[] for i in range(args.domains)]
    cnt =0
    dict_corr_to_mis = {}
    dict_miss_to_ocrr = {}
    for i in range (args.domains):
        for j in range(args.words_per_dom):
            miss = "{}{}".format(i,i) + vocab[cnt] + "{}{}".format(i,i)
            ood_words[i].append(miss)
            dict_corr_to_mis[vocab[cnt]] = miss
            dict_miss_to_ocrr[miss]=vocab[cnt]
            cnt+=1
            
    #print (dict_corr_to_mis)
    #print(dict_miss_to_ocrr)
    #print(ood_words)

    all_ood_words = 0
    all_ood_removed = 0
    all_ood_remain = 0
    all_ood_trans_corr = 0

    dom_ood = [0]*args.domains
    dom_removed=[0]*args.domains
    dom_remain = [0]*args.domains
    dom_trans_corr = [0]*args.domains

    dom_spread_out = [0]*args.domains
    dom_all_out=[0]*args.domains

    id_dict={}
    id_cnt = 0
    with open(args.input_ood, "r") as original_file , open(args.input_translated, "r") as trans_file, open(args.dom_file, "r") as ids_file:
            for i, ( orig, trans, id_dom ) in enumerate( zip(original_file, trans_file, ids_file)):
                id_num = int(id_dom[-2])
                token_list_orig = orig.split()
                token_list_trans = trans.split()
                for element in ood_words[id_num]:
                    if element in token_list_orig:
                        all_ood_words += 1
                        dom_ood[id_num] += 1
                        if element in token_list_trans:
                            #then it's unchanged
                            all_ood_remain +=1
                            dom_remain[id_num] += 1
                        elif dict_miss_to_ocrr[element] in token_list_trans:
                            #then it's corrected
                            all_ood_trans_corr += 1
                            dom_trans_corr[id_num] += 1
                        else:
                            all_ood_removed +=1
                            dom_removed[id_num] += 1
                
                ##Now, we should check the spread
                for k in range (args.domains):
                    if k != id_num: #if it is not from that domain
                        #print(ood_words)
                        for word in ood_words[k]:
                            #print(k)
                            if (dict_miss_to_ocrr[word] in token_list_orig):
                                dom_all_out[k] +=1
                                if word in token_list_trans:
                                    dom_spread_out[k] +=1

    spreads = []

    print("Overall Metrics:")
    print("All ood words: ", all_ood_words)
    print("Corrected ratio: ", float(all_ood_trans_corr)/float(all_ood_words))
    print("Remain ratio: ", float(all_ood_remain)/float(all_ood_words))
    print("Removed ratio: ", float(all_ood_removed)/float(all_ood_words)) 


    print("Domain Metrics: ")   
    for i in range (args.domains):
        print("Results for domain {}".format(i))
        print("\t Corrected ratio: ", float(dom_trans_corr[i])/float(dom_ood[i]))
        print("\t Remain ratio: ", float(dom_remain[i])/float(dom_ood[i]))
        print("\t Removed ratio: ", float(dom_removed[i])/float(dom_ood[i])) 
        print("\t Spread ratio: ", float(dom_spread_out[i])/float(dom_all_out[i])) 
        spreads.append(float(dom_spread_out[i])/float(dom_all_out[i]))

    if args.randm:
      

        print( float(all_ood_trans_corr)/float(all_ood_words)*100.0)
        print( float(all_ood_remain)/float(all_ood_words)*100.0)
        print( float(all_ood_removed)/float(all_ood_words)*100.0) 
        print(sum(spreads)/float(len(spreads))*100.0)

        for i in range (args.domains):
            print( float(dom_trans_corr[i])/float(dom_ood[i]))
            print( float(dom_remain[i])/float(dom_ood[i]))
            print( float(dom_removed[i])/float(dom_ood[i])) 
            print( float(dom_spread_out[i])/float(dom_all_out[i])) 
            
    print("final Overall Metrics:")
    print( float(all_ood_trans_corr)/float(all_ood_words)*100.0)
    print( float(all_ood_remain)/float(all_ood_words)*100.0)
    print( float(all_ood_removed)/float(all_ood_words)*100.0) 
    print(sum(spreads)/float(len(spreads))*100.0)  

    output_file =( args.input_translated)[0]+"_metrics.txt"
    #print("counts")
    #print("all dom 0 words", dom_ood[0])
    #print("all dom 1 words", dom_ood[1])
    #print("all dom 2 words", dom_ood[2])

    #print("counts")
    #print("all dom 0 words", dom_all_out[0])
    #print("all dom 1 words", dom_all_out[1])
    #print("all dom 2 words", dom_all_out[2])


def main(flags=None):
    """Main method for `id_extractor.py`

    Args:
        flags (List[str], optional): command line flags, useful for debugging. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--input_ood", type=str, help="original input with misspellings", required=True)
    parser.add_argument("--input_translated", type=str, help="translation generated by the model", required=True)
    parser.add_argument("--dom_file", type=str, help="domain of each model", required=True)
    
    parser.add_argument("--domains", type=int, help="no domains",  default=3)
    parser.add_argument("--words_per_dom", type=int, help="words per domain", default=5)

    parser.add_argument("--vocab", type=str, help="vocab dictionary of ood words", required=True)
    parser.add_argument("--randm", type=bool, help="random", default=False)
    #parser.add_argument("--output", type=str, help="where to savemetrics generated", default="")

    args = parser.parse_args(flags)
    
    metrics(args)
    #log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "successfully extracted raw ids")

if __name__ == '__main__':
    main()
   
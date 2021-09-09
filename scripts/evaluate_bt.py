import argparse
import random
from collections import defaultdict

def bt(args):
    all_words = 0.0
    corr_words = 0.0
    with open(args.input_src, "r") as input_src , open(args.input_trg, "r") as input_trg:
        for i, ( changed, corr ) in enumerate( zip(input_src, input_trg)):
            for j in range (min(len(corr), len(changed))) :                          
                if changed[j] == corr[j]:
                    corr_words += 1
            all_words += len(changed)
    
    print("accuracy of bt is: ", corr_words/all_words)
   

def main(flags=None):
    """Main method for `id_extractor.py`

    Args:
        flags (List[str], optional): command line flags, useful for debugging. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--input_src", type=str, help="file for encoded data", required=True)
    parser.add_argument("--input_trg", type=str, help="file for raw data", required=True)



 
   

    args = parser.parse_args(flags)
    
    bt(args)
    #log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "successfully extracted raw ids")

if __name__ == '__main__':
    main()
    # example usage: python -m smartcompose_dp.neural_lang_model.lib.preprocessing.id_extractor --input /home/t-famire/json_avocado.json --clean_data /home/t-famire/avocado.dat  --output_encoded_ids /home/t-famire/encoded_ids.txt  --raw_ids=True --output_raw_ids=/home/t-famire/ids.txt

import argparse
import random
from collections import defaultdict

def drop_sentence(args):

    with open(args.input_src, "r") as input_src , open(args.input_trg, "r") as input_trg, open(args.input_docs, "r") as input_docs, open(args.output_src, "w") as output_src, open(args.output_trg, "w") as output_trg, open(args.output_docs, "w") as output_docs:
        for i, ( sent, dom, doc ) in enumerate( zip(input_src, input_trg, input_docs)):
            if (int(doc[:-1])%args.drop  == 0):                           
                output_src.write(sent)
                output_trg.write(dom)
                output_docs.write(doc)
   

def main(flags=None):
    """Main method for `id_extractor.py`

    Args:
        flags (List[str], optional): command line flags, useful for debugging. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--input_src", type=str, help="file for encoded data", required=True)
    parser.add_argument("--input_trg", type=str, help="file for raw data", required=True)
    parser.add_argument("--input_docs", type=str, help="file for docs data", required=True)



    parser.add_argument("--output_src", type=str, help="subsampled txt file", required=True)
    parser.add_argument("--output_trg", type=str, help="subsampled trg file", required=True)
    parser.add_argument("--output_docs", type=str, help="subsampled trg file", required=True)


    parser.add_argument("--drop", type=float, help="what ratio to drop", default=20)
 
   

    args = parser.parse_args(flags)
    
    drop_sentence(args)
    #log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "successfully extracted raw ids")

if __name__ == '__main__':
    main()
    # example usage: python -m smartcompose_dp.neural_lang_model.lib.preprocessing.id_extractor --input /home/t-famire/json_avocado.json --clean_data /home/t-famire/avocado.dat  --output_encoded_ids /home/t-famire/encoded_ids.txt  --raw_ids=True --output_raw_ids=/home/t-famire/ids.txt

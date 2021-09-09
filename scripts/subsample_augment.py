import argparse
import random
from collections import defaultdict

def drop_sentence(args):

    with open(args.input_src, "r") as input_src, open(args.output_src, "w") as output_src, open(args.output_trg, "w") as output_trg:
        for i, ( sent ) in enumerate( input_src):
            if (i%args.drop  == 0):                           
                output_src.write(sent)
                output_trg.write('z\n')

   

def main(flags=None):
    """Main method for `id_extractor.py`

    Args:
        flags (List[str], optional): command line flags, useful for debugging. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--input_src", type=str, help="file for encoded data", required=True)



    parser.add_argument("--output_src", type=str, help="subsampled txt file", required=True)
    parser.add_argument("--output_trg", type=str, help="subsampled trg file", required=True)


    parser.add_argument("--drop", type=float, help="what ratio to drop", default=5)
 
   

    args = parser.parse_args(flags)
    
    drop_sentence(args)
    #log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "successfully extracted raw ids")

if __name__ == '__main__':
    main()
    # example usage: python -m smartcompose_dp.neural_lang_model.lib.preprocessing.id_extractor --input /home/t-famire/json_avocado.json --clean_data /home/t-famire/avocado.dat  --output_encoded_ids /home/t-famire/encoded_ids.txt  --raw_ids=True --output_raw_ids=/home/t-famire/ids.txt

import argparse


def lex_div(args):
# Open a file: file
    file = open(args.input,mode='r')
    word_set = set()
    word_list = []
    for line in file:
        for word in line.split():
            word_list.append(word)
            word_set.add(word)
    # read all lines at once
    
    # close the file
    file.close()
    print(len(word_set))
    print(len(word_set)/len(word_list))
   
    
   

def main(flags=None):
    """Main method for `id_extractor.py`

    Args:
        flags (List[str], optional): command line flags, useful for debugging. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--input", type=str, help="file for  data", required=True)
  

 
   

    args = parser.parse_args(flags)
    
    lex_div(args)
    #log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "successfully extracted raw ids")

if __name__ == '__main__':
    main()
    # example usage: python -m smartcompose_dp.neural_lang_model.lib.preprocessing.id_extractor --input /home/t-famire/json_avocado.json --clean_data /home/t-famire/avocado.dat  --output_encoded_ids /home/t-famire/encoded_ids.txt  --raw_ids=True --output_raw_ids=/home/t-famire/ids.txt

import argparse

def get_confusion_mat(args):
    #mat =[0]*args.no_styles
    mat_full = [ [0]*args.no_styles for _ in range(args.no_styles) ]#[[0]*args.no_styles]*args.no_styles
    print (mat_full)
    all_real_cnt = [0] * args.no_styles
    all_pred_cnt = [0] * args.no_styles
    with open(args.input_preds, "r") as preds_file, open(args.input_trgs, "r") as trgs_file:
        for i, ( pred, trg ) in enumerate( zip(preds_file, trgs_file)):
            #print(int(pred[0]), int(trg[0]))
            mat_full[int(trg[0])][int(pred[0])] += 1
            all_real_cnt[int(trg)] += 1
            all_pred_cnt[int(pred)] += 1
            

    print(mat_full)
    print(all_pred_cnt)
    print(all_real_cnt)
    print(sum(all_pred_cnt))
    print(sum(all_real_cnt))
    print(sum(mat_full[0][:]))
    print(sum(mat_full[1][:]))
    print(sum(mat_full[2][:]))
    print("*****")
    for i,row in enumerate(mat_full):
        print(row[0]/all_real_cnt[i], row[1]/all_real_cnt[i], row[2]/all_real_cnt[i])


def main(flags=None):
    """Main method for `id_extractor.py`

    Args:
        flags (List[str], optional): command line flags, useful for debugging. Defaults to None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument("--input_preds", type=str, help="file for classifier prediction", required=True)
    parser.add_argument("--input_trgs", type=str, help="file for classifier ", required=True)
    parser.add_argument("--no_styles", type=int, help="file for classifier ", default=3)

 
   

    args = parser.parse_args(flags)
    
    get_confusion_mat(args)
    #log(logging.INFO, DataCategory.ONLY_PUBLIC_DATA, "successfully extracted raw ids")

if __name__ == '__main__':
    main()
    # example usage: python -m smartcompose_dp.neural_lang_model.lib.preprocessing.id_extractor --input /home/t-famire/json_avocado.json --clean_data /home/t-famire/avocado.dat  --output_encoded_ids /home/t-famire/encoded_ids.txt  --raw_ids=True --output_raw_ids=/home/t-famire/ids.txt

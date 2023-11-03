import os
import json
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, required=True, help="the directory which stores the image tsvfiles and the text jsonl annotations"
    )
    parser.add_argument(
        "--splits", type=str, required=True, help="specify the dataset splits which this script processes, concatenated by comma \
            (e.g. train,valid,test)"
    )
    return parser.parse_args()


def count_sequence_length():
    args = parse_args()
    # read specified dataset splits
    specified_splits = list(set(args.splits.strip().split(",")))
    print("Dataset splits to be processed: {}".format(", ".join(specified_splits)))

    for split in specified_splits:
        pairs_annotation_path = os.path.join(args.data_dir, "MR_{}_queries.jsonl".format(split))
        with open(pairs_annotation_path, "r", encoding="utf-8") as fin_pairs:
            max_seq_len = 0
            for line in tqdm(fin_pairs):
                line = line.strip()
                obj = json.loads(line)

                max_seq_len = len(obj['query_text']) if len(obj['query_text']) > max_seq_len else max_seq_len
        
        print("max seq len of {}: {}".format(split, max_seq_len))



if __name__ == "__main__":
    count_sequence_length()
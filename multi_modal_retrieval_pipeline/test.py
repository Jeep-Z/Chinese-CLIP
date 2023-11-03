import json
from tqdm import tqdm
from pathlib import Path


img_feat_path = Path("/workdir/Chinese-CLIP/data/datasets/Multimodal_Retrieval/MR_test_queries.jsonl.img_feat.jsonl")
test_feat_path = Path("/workdir/Chinese-CLIP/data/datasets/Multimodal_Retrieval/MR_test_queries.txt_feat.jsonl")

with open(img_feat_path, "r", encoding="utf-8") as img_f:
    for line in tqdm(img_f):
        line = line.strip()
        obj = json.loads(line)
        print(obj)
        break

with open(test_feat_path, "r", encoding="utf-8") as txt_f:
    for line in tqdm(txt_f):
        line = line.strip()
        obj = json.loads(line)
        print(obj)
        break
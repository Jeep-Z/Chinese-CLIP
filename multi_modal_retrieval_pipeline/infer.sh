split=test # 指定计算valid或test集特征
DATAPATH=./data
dataset_name=Multimodal_Retrieval
python -u multi_modal_retrieval_pipeline/make_topk_predictions.py \
    --image-feats="${DATAPATH}/datasets/${dataset_name}/MR_${split}_imgs.img_feat.jsonl" \
    --text-feats="${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.txt_feat.jsonl" \
    --top-k=10 \
    --eval-batch-size=32768 \
    --output="${DATAPATH}/datasets/${dataset_name}/${split}_predictions.jsonl"

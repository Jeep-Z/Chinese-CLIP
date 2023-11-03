export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

split=test # 指定计算valid或test集特征
DATAPATH=./data
resume=${DATAPATH}/experiments/Multimodal_Retrieval_finetune_vit-b-16_roberta-base_bs16_1gpu/checkpoints/epoch_latest.pt
dataset_name=Multimodal_Retrieval

python -u multi_modal_retrieval_pipeline/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.jsonl" \
    --img-batch-size=32 \
    --text-batch-size=32 \
    --context-length=30 \
    --resume=${resume} \
    --vision-model=ViT-B-16 \
    --text-model=RoBERTa-wwm-ext-base-chinese
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip

split=test # 指定计算valid或test集特征
DATAPATH=./data
resume=${DATAPATH}/pretrained_weights/clip_cn_vit-h-14.pt
dataset_name=Multimodal_Retrieval

python -u multi_modal_retrieval_pipeline/extract_features.py \
    --extract-image-feats \
    --extract-text-feats \
    --image-data="${DATAPATH}/datasets/${dataset_name}/lmdb/${split}/imgs" \
    --text-data="${DATAPATH}/datasets/${dataset_name}/MR_${split}_queries.jsonl" \
    --img-batch-size=128 \
    --text-batch-size=128 \
    --context-length=30 \
    --resume=${resume} \
    --vision-model=ViT-H-14 \
    --text-model=RoBERTa-wwm-ext-large-chinese

#ViT-B-16
#RoBERTa-wwm-ext-base-chinese
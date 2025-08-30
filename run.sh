
vis_feat='clip'
text_feat='clip'
context_encoder='bilstm'
run_id='1'
lr=1e-5
batch_size=16
dataset='kit'
epoch=50
echo "===================================================="
echo $run_id " | " "train" " | " $vis_feat " | " $text_feat " | " $context_encoder " | " $dataset " | " $lr " | " $batch_size " | " $epoch
echo "===================================================="
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py --split_type 'train' --sample_rate 2 --run_id $run_id --fp_seg 20 --visual_feature_extractor $vis_feat --text_feature_extractor $text_feat --context_encoder $context_encoder --dataset $dataset --lr $lr --batch_size $batch_size --epoch $epoch 

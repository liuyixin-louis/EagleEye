###### 1. Search ######
python3 search.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--checkpoint models/ckpt/imagenet_mobilenet_full_model.pth \
--gpu_ids 2 \
--batch_size 8 \
--dataset_path /data/imagenet \
--dataset_name imagenet_train_val_split \
--num_workers 4 \
--flops_target 0.5 \
--max_rate 0.7 \
--affine 0 \
--output_file search_results/mbv1_strategies.txt \
--compress_schedule_path compress_config/mbv1_imagenet.yaml

##### 2. Selection #######
python choose_strategy.py search_results/mbv1_strategies.txt

##### 3. Fine-tuning #######
python3 finetune.py \
--model_name mobilenetv1 \
--num_classes 1000 \
--checkpoint models/ckpt/imagenet_mobilenet_full_model.pth \
--gpu_ids [GPU_IDS] \
--batch_size 512 \
--dataset_path /data/imagenet \
--dataset_name imagenet \
--exp_name mbv1_50flops \
--search_result search_results/mbv1_strategies.txt \
--strategy_id 0 \
--epoch 120 \
--lr 1e-2 \
--weight_decay 1e-4

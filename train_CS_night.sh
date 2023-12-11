#!/bin/bash
# changeable
random_seed=1 # 1, 12, 123, 1234, 12345
source_dataset="cityscapes"
path_to_source_dataset="/nfs/s3_common_dataset/cityscapes"
target_dataset="ACDC"
path_to_target_dataset="/nfs/ofs-902-1/object-detection/jiangjing/datasets/ACDC"
ACDC_sub="night"
target_domain_description="driving at night"

# unchanged
path_to_source_checkpoint_dir="/nfs/ofs-902-1/object-detection/jiangjing/experiments/PODA/saved_ckpts/seed_${random_seed}/${source_dataset}_${target_dataset}_${ACDC_sub}_source"
path_to_source_checkpoint="${path_to_source_checkpoint_dir}/best_deeplabv3plus_resnet_clip_${source_dataset}.pth"
path_to_target_checkpoint_dir="/nfs/ofs-902-1/object-detection/jiangjing/experiments/PODA/saved_ckpts/seed_${random_seed}/${source_dataset}_${target_dataset}_${ACDC_sub}_target"
path_to_target_checkpoint="${path_to_target_checkpoint_dir}/adapted_deeplabv3plus_resnet_clip_${source_dataset}.pth"
directory_for_saved_statistics="/nfs/ofs-902-1/object-detection/jiangjing/experiments/PODA/saved_statistics/seed_${random_seed}/${source_dataset}_${target_dataset}"

#echo "---step1 Source training---"
## changed val_interval to 10000 for faster train
#CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python main.py \
#  --dataset ${source_dataset} \
#  --data_root ${path_to_source_dataset} \
#  --data_aug \
#  --lr 0.1 \
#  --crop_size 768 \
#  --batch_size 2 \
#  --freeze_BB \
#  --val_interval 10000 \
#  --random_seed ${random_seed} \
#  --ACDC_sub "${ACDC_sub}" \
#  --ckpts_path ${path_to_source_checkpoint_dir}
#
#echo "---step2 Feature optimization---"
#CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python PIN_aug.py \
#  --dataset ${source_dataset} \
#  --data_root ${path_to_source_dataset} \
#  --total_it 100 \
#  --resize_feat \
#  --domain_desc "${target_domain_description}" \
#  --random_seed ${random_seed} \
#  --ACDC_sub "${ACDC_sub}" \
#  --save_dir ${directory_for_saved_statistics}
#
#echo "---step3 Model adaptation---"
#echo "loading statistics_dir [${directory_for_saved_statistics}]"
#echo "loading source model [${path_to_source_checkpoint}]"
#CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python main.py \
#  --dataset ${source_dataset} \
#  --data_root ${path_to_source_dataset} \
#  --ckpt ${path_to_source_checkpoint} \
#  --batch_size 8 \
#  --lr 0.01 \
#  --ckpts_path ${path_to_target_checkpoint_dir} \
#  --freeze_BB \
#  --train_aug \
#  --total_itrs 2000 \
#  --random_seed ${random_seed} \
#  --ACDC_sub "${ACDC_sub}" \
#  --path_mu_sig ${directory_for_saved_statistics}

#echo "---starting test---"
#echo "loading adapted_target model [${path_to_target_checkpoint}]"
#CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python main.py \
#  --dataset ${target_dataset} \
#  --data_root ${path_to_target_dataset} \
#  --ckpt ${path_to_target_checkpoint} \
#  --test_only \
#  --random_seed ${random_seed} \
#  --ACDC_sub "${ACDC_sub}" \
#  --val_batch_size 1

# 暂时用一下的，用来测试source-only
target_dataset="gta5"
path_to_target_dataset="/nfs/s3_common_dataset/GTAV"
ACDC_sub="GTA5"
echo "---starting test---"
echo "loading adapted_target model [${path_to_target_checkpoint}]"
CUDA_VISIBLE_DEVICES=0 /home/luban/apps/miniconda/miniconda/envs/torch1101/bin/python main.py \
  --dataset ${target_dataset} \
  --data_root ${path_to_target_dataset} \
  --ckpt ${path_to_source_checkpoint} \
  --test_only \
  --random_seed ${random_seed} \
  --ACDC_sub "${ACDC_sub}" \
  --val_batch_size 1

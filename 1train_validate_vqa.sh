#!/usr/bin/env bash

python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_cvpr_check.yaml GPU_ID 2
python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_edited_train.yaml VAL.SPLIT_VQA val2014_10 GPU_ID 2
python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_edited_train.yaml VAL.SPLIT_VQA train2014 GPU_ID 2
python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_edited_train.yaml VAL.SPLIT_VQA edited_train2014 GPU_ID 2



python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_edited_train.yaml GPU_ID 1
python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_edited_train.yaml VAL.SPLIT_VQA val2014_10 GPU_ID 1
python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_edited_train.yaml VAL.SPLIT_VQA train2014 GPU_ID 1
python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_train_edited_train.yaml VAL.SPLIT_VQA edited_train2014 GPU_ID 1

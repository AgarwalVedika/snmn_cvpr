#!/usr/bin/env bash
#python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_finetune_is_this_a_orig_10_25lr.yaml TRAIN.SOLVER.LR 0.00025 TRAIN.INIT_FROM_WEIGHTS True TRAIN.INIT_WEIGHTS_FILE ./exp_vqa/tfmodel/vqa_v2_scratch_train_25lr/00000030 GPU_ID 0
#python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_finetune_is_this_a_orig_10_25lr.yaml GPU_ID 0
#python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_finetune_is_this_a_orig_all_25lr.yaml TRAIN.SOLVER.LR 0.00025 TRAIN.INIT_FROM_WEIGHTS True TRAIN.INIT_WEIGHTS_FILE ./exp_vqa/tfmodel/vqa_v2_scratch_train_25lr/00000030 GPU_ID 0
#python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_finetune_is_this_a_orig_all_25lr.yaml GPU_ID 0
#python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_finetune_is_this_a_orig_10_edited10_25lr.yaml TRAIN.SOLVER.LR 0.00025 TRAIN.INIT_FROM_WEIGHTS True TRAIN.INIT_WEIGHTS_FILE ./exp_vqa/tfmodel/vqa_v2_scratch_train_25lr/00000030 GPU_ID 0
#python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_finetune_is_this_a_orig_10_edited10_25lr.yaml GPU_ID 0


python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/vqa_v2_scratch_finetune_what_color_is_the_orig_all_edited10_25lr.yaml TRAIN.SOLVER.LR 0.00025 TRAIN.INIT_FROM_WEIGHTS True TRAIN.INIT_WEIGHTS_FILE ./exp_vqa/tfmodel/vqa_v2_scratch_train_25lr/00000030 GPU_ID 0
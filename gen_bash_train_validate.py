import os
import json
import ipdb
import argparse

## for train val plots: do this:


# ques_type_filter = ['counting'; 'is there a' ; 'is this a' ; 'what color is the' ; 'how many' ]
# token = ['all_ans_same', 'everyIQA']
# saving_imdb_prefix = 'original_' + token + '_' + ques_type_filter
# mode = ['val2014']

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id_sh', required=True, type=int)   ## can only be original
args = parser.parse_args()



## yaml files
exp_list= ['vqa_v2_scratch_train1_0001lr', 'vqa_v2_scratch_train1_25lr', 'vqa_v2_scratch_train1_75_e_5lr']
#val_splits = ['train2014',  'val2014']



ques_type_filter = ['how_many', 'is_there_a', 'is_this_a', 'what_color_is_the', 'counting']

idx = 4
ques_type = ques_type_filter[idx]
print(ques_type)


## yaml files
yaml_prefix0 = 'vqa_v2_scratch_'
yaml_prefix = 'vqa_v2_scratch_finetune_'   # vqa_v2_scratch_finetune_counting_2_orig_10_edited10_25lr



possible7_models = ['orig_10', 'orig_10_del1', 'orig_10_del1_edited10', 'orig_10_edited10', 'orig_all', 'orig_all_del1', 'orig_all_del1_edited10', 'orig_all_del1_edited_all']

#possible7_models = [ 'orig_10', 'orig_all', 'orig_10_edited10', 'orig_all_edited10']


yaml_suffix = '_25lr'

train_solver_lr = 0.00025
init_from_weights = 'True'
model_init =




with open( 'all' + ques_type+ '_25lr_4_validate_vqa.sh', 'w+') as file:
    for yaml_file_crux in possible7_models:
        if ques_type == 'counting':
            exp_name = yaml_prefix + ques_type + '_2_' + yaml_file_crux + yaml_suffix
        else:
            exp_name = yaml_prefix + ques_type + '_' + yaml_file_crux + yaml_suffix
        file.write('python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/{}.yaml TRAIN.SOLVER.LR {} TRAIN.INIT_FROM_WEIGHTS {} TRAIN.INIT_WEIGHTS_FILE {} GPU_ID {}'.format
                          (exp_name, train_solver_lr,  init_from_weights, model_init ,args.gpu_id_sh) + "\n")

        file.write('python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/{}.yaml VAL.SPLIT_VQA {} GPU_ID {}'.format
                       (exp_name,   ques_type + '_val2014_10_10'   ,args.gpu_id_sh) + "\n")
        file.write('python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/{}.yaml VAL.SPLIT_VQA {} GPU_ID {}'.format
                       (exp_name,   ques_type + '_val2014_10'   ,args.gpu_id_sh) + "\n")

# idx=2
# exp_list= ['vqa_v2_scratch_train_edited_train_25lr', 'vqa_v2_scratch_train_edited_train' , 'vqa_v2_scratch_train']
# exp_name = exp_list[idx]
# with open( str(idx) + 'train_validate_vqa.sh', 'w+') as file:
#         file.write('python ./exp_vqa/train_net_vqa_epoch.py --cfg exp_vqa/cfgs/{}.yaml GPU_ID {}'.format
#                           (exp_name, args.gpu_id_sh) + "\n")
#         ## this is validation- validation
#         file.write('python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/{}.yaml VAL.SPLIT_VQA {} GPU_ID {}'.format
#                        (exp_name,   'val2014_10',args.gpu_id_sh) + "\n")
#
#         ## these 2 below corresponds to train splits
#         file.write('python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/{}.yaml VAL.SPLIT_VQA {} GPU_ID {}'.format
#                        (exp_name,   'train2014'  ,args.gpu_id_sh) + "\n")
#         file.write('python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/{}.yaml VAL.SPLIT_VQA {} GPU_ID {}'.format
#                        (exp_name,   'edited_train2014',args.gpu_id_sh) + "\n")




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
exp_list= ['vqa_v2_scratch_train_0001lr', 'vqa_v2_scratch_train_25lr', 'vqa_v2_scratch_train_50lr', 'vqa_v2_scratch_train_75lr']
#val_splits = ['train2014',  'val2014']
exp_list_2= ['vqa_v2_scratch_train1_75_e_5lr']

with open('run_validate_validate_vqa_75lr.sh', 'w+') as file:
    for exp_name in exp_list_2:
        file.write('python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/{}.yaml VAL.SPLIT_VQA {} GPU_ID {}'.format
                       (exp_name, 'train2014', args.gpu_id_sh) + "\n")
        file.write('python ./exp_vqa/validate_net_epoch.py --cfg exp_vqa/cfgs/{}.yaml VAL.SPLIT_VQA {} GPU_ID {}'.format
                       (exp_name, 'val2014', args.gpu_id_sh) + "\n")



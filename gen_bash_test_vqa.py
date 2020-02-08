import os
import json
import ipdb
import argparse


# ques_type_filter = ['counting'; 'is there a' ; 'is this a' ; 'what color is the' ; 'how many' ]
# token = ['all_ans_same', 'everyIQA']
# saving_imdb_prefix = 'original_' + token + '_' + ques_type_filter
# mode = ['val2014']

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id_sh', required=True, type=int)   ## can only be original
args = parser.parse_args()




ques_type_filter = ['how_many', 'is_there_a', 'is_this_a', 'what_color_is_the', 'counting']
idx = 3
ques_type = ques_type_filter[idx]
print(ques_type)


## yaml files
yaml_prefix0 = 'vqa_v2_scratch_'
yaml_prefix = 'vqa_v2_scratch_finetune_' + ques_type + '_'   # vqa_v2_scratch_finetune_counting_2_orig_10_edited10_25lr


possible_models = ['train', 'orig_10', 'orig_all', 'orig_10_edited10']
splits_what_color = ['train', 'orig_10', 'orig_all', 'orig_10_edited10',  'orig_all_edited10']
splits_counting = ['train', 'orig_10', 'orig_all', 'orig_10_del1_edited10', 'orig_10_edited10']

if ques_type=='counting':
    yaml_prefix = 'vqa_v2_scratch_finetune_counting_2_'
    possible_models = splits_counting

if ques_type=='what_color_is_the':
    possible_models = splits_what_color

yaml_suffix = '_25lr.yaml'

# test_splits = [ 'counting_val2014_90_10' , 'edited_all_ans_same_counting_val2014', 'counting_del1_edited_val2014' , 'counting_val2014_90' ,  'edited_everyIQA_counting_val2014']

test_splits = [ ques_type +'_val2014_90_10' , 'edited_all_ans_same_' + ques_type +'_val2014',  ques_type +'_val2014_90' ,  'edited_everyIQA_' + ques_type +'_val2014']
with open(ques_type +'_test_run_test_vqa.sh', 'w+') as file:
    for yaml_file_crux in possible_models:
        if yaml_file_crux == 'train' :#or yaml_file_crux == 'train_edited_train':
            #yaml_file = yaml_prefix0 + yaml_file_crux + yaml_suffix
            yaml_file = 'vqa_v2_scratch_train_25lr.yaml'#'vqa_v2_scratch.yaml'   # epoch 30: gave the best accuracy of 44%
        else:
            yaml_file = yaml_prefix + yaml_file_crux + yaml_suffix
        for test in test_splits:
            file.write('python /BS/vedika2/nobackup/snmn/exp_vqa/test_net_vqa_without_dump.py --cfg exp_vqa/cfgs/{} TEST.SPLIT_VQA {} GPU_ID {}'.format
                           (yaml_file, test, args.gpu_id_sh) + "\n")



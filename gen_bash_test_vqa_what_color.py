import os
import json
import ipdb
import argparse


# ques_type_filter = ['counting'; 'is there a' ; 'is this a' ; 'what color is the' ; 'how many' ]
# token = ['all_ans_same', 'everyIQA']
# saving_imdb_prefix = 'original_' + token + '_' + ques_type_filter
# mode = ['val2014']

# parser = argparse.ArgumentParser()
# parser.add_argument('--gpu_id_sh', required=True, type=int)   ## can only be original
# args = parser.parse_args()


#
#
# ques_type = 'what_color_is_the'
# ## yaml files
# yaml_prefix0 = 'vqa_v2_scratch_'
# yaml_prefix = 'vqa_v2_scratch_finetune_' + ques_type + '_'   # vqa_v2_scratch_finetune_counting_2_orig_10_edited10_25lr
#
#
# splits_what_color = [ 'vqa_v2_scratch_finetune_what_color_is_the_orig_10_edited10_25lr_ratio_naive_70_30.yaml']  ## train is the epoch 00 answer; orig_10: baseline finetunied using only orig; rest using orig+edit
# if ques_type=='what_color_is_the':
#     possible_models = splits_what_color
#
# test_splits = [ ques_type +'_val2014_90_10' , 'edited_all_ans_same_' + ques_type +'_val2014']
#
#
# with open(ques_type +'_test_run_test_vqa2.sh', 'w+') as file:
#     for yaml_file in possible_models:
#         for test in test_splits:
#             for iter_num_epoch in range(1,51,1):
#                 file.write(
#                     'python /BS/vedika2/nobackup/snmn/exp_vqa/test_net_vqa.py --cfg exp_vqa/cfgs/{} TEST.SPLIT_VQA {} TEST.ITER {} GPU_ID {}'.format
#                     (yaml_file, test, iter_num_epoch, args.gpu_id_sh) + "\n")
#
#
# #
#
# ques_type = 'counting'
# #possible_models = [ 'vqa_v2_scratch_finetune_counting_2_orig_10_del1_25lr.yaml']  ## train is the epoch 00 answer; orig_10: baseline finetunied using only orig; rest using orig+edit
# #possible_models = [ 'vqa_v2_scratch_finetune_counting_2_orig_10_del1_edited10_25lr.yaml']#,
# possible_models = ['vqa_v2_scratch_finetune_counting_2_orig_10_25lr.yaml']
# test_splits = [ 'counting_val2014_90_10', 'counting_del1_edited_val2014']
# test_splits = [ 'counting_val2014_10_10']
#
# with open(ques_type +'_test_run_test_vqa3.sh', 'w+') as file:
#     for yaml_file in possible_models:
#         for test in test_splits:
#             for iter_num_epoch in range(1,51,1):
#                 file.write(
#                     'python /BS/vedika2/nobackup/snmn/exp_vqa/test_net_vqa.py --cfg exp_vqa/cfgs/{} TEST.SPLIT_VQA {} TEST.ITER {} GPU_ID {}'.format
#                     (yaml_file, test, iter_num_epoch, args.gpu_id_sh) + "\n")



# ques_types = ['what_color_is_the', 'how_many', 'is_there_a', 'is_this_a']
# ques_types = ['counting']
# for ques_type in ques_types:
#     if ques_type =='counting':
#         possible_models = ['vqa_v2_scratch_finetune_' + ques_type + '_2_orig_10_edited10_25lr.yaml',
#                            'vqa_v2_scratch_finetune_' + ques_type + '_2_orig_10_25lr.yaml']
#     else:
#         possible_models = [ 'vqa_v2_scratch_finetune_'+ ques_type + '_orig_10_edited10_25lr.yaml', 'vqa_v2_scratch_finetune_'+ ques_type + '_orig_10_25lr.yaml']
#     test_splits = [ 'edited_all_ans_same_' + ques_type +'_val2014']
#     #test_splits = [ ques_type +'_val2014_10_10']
#     #test_splits = [ ques_type + '_val2014_90_10']
#     with open(ques_type +'_test_run_test_vqa3.sh', 'w+') as file:
#         for yaml_file in possible_models:
#             for test in test_splits:
#                 for iter_num_epoch in range(1,51,1):
#                     file.write(
#                         'python /BS/vedika2/nobackup/snmn/exp_vqa/test_net_vqa.py --cfg exp_vqa/cfgs/{} TEST.SPLIT_VQA {} TEST.ITER {} GPU_ID {}'.format
#                         (yaml_file, test, iter_num_epoch, 1) + "\n")




 
possible_models = [  'vqa_v2_scratch_train_edited_train_25lr_finetuning.yaml'   ] #,  ]   'vqa_v2_scratch_train_edited_train_25lr.yaml'
test_splits = [  'val2014_10' ]  # 'val2014_10' ,'edited_val2014',
#test_splits = [ ques_type +'_val2014_10_10']
#test_splits = [ ques_type + '_val2014_90_10']

with open('test_run_test_vqa_ft.sh', 'w+') as file:
    ipdb.set_trace()
    for yaml_file in possible_models:
        for test in test_splits:
            for iter_num_epoch in range(38,45,1):
                file.write(
                    'python /BS/vedika2/nobackup/snmn/exp_vqa/test_net_vqa.py --cfg exp_vqa/cfgs/{} TEST.SPLIT_VQA {} TEST.ITER {} GPU_ID {}'.format
                    (yaml_file, test, iter_num_epoch, 0) + "\n")
                    
#possible_models = [  'vqa_v2_scratch_train_cvpr_check.yaml'   ] #,  ]   'vqa_v2_scratch_train_edited_train_25lr.yaml'
#test_splits = [ 'val2014', 'edited_val2014']
##test_splits = [ ques_type +'_val2014_10_10']
##test_splits = [ ques_type + '_val2014_90_10']

#with open('test_run_test_vqa_cvpr_check.sh', 'w+') as file:
    #ipdb.set_trace()
    #for yaml_file in possible_models:
        #for test in test_splits:
            #for iter_num_epoch in range(1,51,1):
                #file.write(
                    #'python /BS/vedika2/nobackup/snmn/exp_vqa/test_net_vqa.py --cfg exp_vqa/cfgs/{} TEST.SPLIT_VQA {} TEST.ITER {} GPU_ID {}'.format
                    #(yaml_file, test, iter_num_epoch, 0) + "\n")                    

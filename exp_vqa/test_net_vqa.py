import argparse
import os
import json
import numpy as np
import tensorflow as tf
import ipdb
import pickle

from models_vqa.model import Model
from models_vqa.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.vqa_train.data_reader import DataReader

import time
start_time = time.time()
## set seeds
import random
random.seed(1234)           #TODO bhai set random in DataReader
tf.set_random_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
assert cfg.EXP_NAME == os.path.basename(args.cfg).replace('.yaml', '')
if args.opts:
    merge_cfg_from_list(args.opts)

# Start session
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   ### edit_vedika
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
#print(imdb_file)
data_reader = DataReader(
    imdb_file, shuffle=False, one_pass=True, batch_size=cfg.TRAIN.BATCH_SIZE,               ## cfg.TRAIN.BATCH_SIZE =128 (vqa_v2_scratch.yaml)
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=True,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL)
num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
num_choices = data_reader.batch_loader.answer_dict.num_vocab
module_names = data_reader.batch_loader.layout_dict.word_list

# Eval files
if cfg.TEST.GEN_EVAL_FILE:
    eval_file_pkl = cfg.TEST.EVAL_FILE_pkl % (
        cfg.EXP_NAME, cfg.TEST.SPLIT_VQA, cfg.EXP_NAME, cfg.TEST.ITER)
    # print('evaluation outputs will be saved to %s' % eval_file_pkl)
    os.makedirs(os.path.dirname(eval_file_pkl), exist_ok=True)

    answer_word_list = data_reader.batch_loader.answer_dict.word_list      ##aanswer_word_list is here
    assert(answer_word_list[0] == '<unk>')    ### here asserting unk to position 0

# Inputs and model
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, cfg.MODEL.FEAT_DIM])
model = Model(
    input_seq_batch, seq_length_batch, image_feat_batch, num_vocab=num_vocab,
    num_choices=num_choices, module_names=module_names, is_training=False)

# Load snapshot
if cfg.TEST.USE_EMV:
    ema = tf.train.ExponentialMovingAverage(decay=0.9)  # decay doesn't matter
    var_names = {
        (ema.average_name(v) if v in model.params else v.op.name): v
        for v in tf.global_variables()}
else:
    var_names = {v.op.name: v for v in tf.global_variables()}
snapshot_file = cfg.TEST.SNAPSHOT_FILE % (cfg.EXP_NAME, cfg.TEST.ITER) ####they ar eloading the snapshot here- chose the one that does the best
# print('snapshot file', snapshot_file)
# print(cfg.TEST.ITER)
snapshot_saver = tf.train.Saver(var_names)
snapshot_saver.restore(sess, snapshot_file)

# Write results
result_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, cfg.TEST.ITER)
# print(result_dir)
vis_dir = os.path.join(
    result_dir, 'vqa_%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_VQA))
os.makedirs(result_dir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)

# Run test
answer_correct, num_questions = 0, 0

output_qids_answers = []

## np_pkl file style
# file_data = {}
#abcd = []
# all_qid = []
# all_gt_labels_random = []
# all_ans_pred = []
# all_ss_vc = []

for n_batch, batch in enumerate(data_reader.batches()):
    # ipdb> batch.keys()
    #dict_keys(['input_seq_batch', 'seq_length_batch', 'image_feat_batch', 'image_path_list', 'qid_list', 'qstr_list', 'answer_label_batch', 'valid_answers_list', 'all_answers_list', 'gt_layout_batch'])

    fetch_list = [model.vqa_scores]                     ### vqa_scores = tf.nn.softmax(vqa_scores)
    answer_incorrect = num_questions - answer_correct
    fetch_list_val = sess.run(fetch_list, feed_dict={              ## fetch_list_val = [array.shape[128,3001]  is a list with one element that is the array
            input_seq_batch: batch['input_seq_batch'],
            seq_length_batch: batch['seq_length_batch'],
            image_feat_batch: batch['image_feat_batch']})

    # compute accuracy
    vqa_scores_val = fetch_list_val[0]     # ## fetch_list_val = [array.shape[128,3001]  is a list with one element that is the array, hence fetch_list_val[0] gives you the array
    vqa_scores_val[:, 0] = 0   # DESIRED SOFTMAX VECTOR #-1e10  # remove <unk> answer   ## author's## change by vedika- svaing 1e-10 doesnt make sense

    #vqa_pred_max_softmax_scores = np.max(vqa_scores_val, axis=1)             ## edit vedika: getting max model.vqa_scores
    #vqa_pred_max_softmax_scores = vqa_pred_softmax_scores.astype(np.float64)
    vqa_predictions = np.argmax(vqa_scores_val, axis=1)    ### argmax = > getting the index with highest softmax score # vqa_prediction is the label_id

    if data_reader.batch_loader.load_answer:
        vqa_labels = batch['answer_label_batch']            ## gt labels  ids basically- 0 to 30000
    else:
        # dummy labels with all -1 (so accuracy will be zero)
        vqa_labels = -np.ones(vqa_scores_val.shape[0], np.int32)                  ### dummy label given to take care of test set

    score_list =  [np.sum(vqa_predictions[i] == vqa_labels[i]) for i in range(vqa_predictions.shape[0])]
    score_list2 = [1 if i > 2 else i / 3 for i in score_list]
    #answer_correct += np.sum(vqa_predictions == vqa_labels)  #answer_correct += np.sum(vqa_predictions == vqa_labels)  ORIGINALLY
    answer_correct += np.sum(score_list2)
    num_questions += len(vqa_labels)
    accuracy = answer_correct / num_questions

    # if n_batch % 20 == 0:
    #     print('exp: %s, iter = %d, accumulated accuracy on %s = %f (%d / %d)' %
    #           (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
    #            accuracy, answer_correct, num_questions))

    ## vqa_labels, predictions, vqa_scores_val: all are arrays
    ## np_pkl style
    # if cfg.TEST.GEN_EVAL_FILE:
    #     qid_list = batch['qid_list']
    #     all_qid.append(qid_list)
    #     all_gt_labels_random.append(vqa_labels)
    #     all_ans_pred.append(vqa_predictions)
    #     all_ss_vc.append(vqa_scores_val)

    ## old style preferred- easier to read and analyze plus pickl instead of json- as it has no issue storing python objects
    if cfg.TEST.GEN_EVAL_FILE:
        #ipdb.set_trace()
        img_id_list = batch['image_id_list']  ### always a string here
        qid_list = batch['qid_list']
        output_qids_answers += [
            {'ques_id': qid, 'img_id': iid, 'ans_id': p,  'gt_ans_id_used': gt_label, 'ss_vc': softmax_vector}  ## int(qid); softmax_vector.tolist()  because json does not recognize NumPy data types. Convert the number to a Python int before serializing the object:
            for qid, iid, p, gt_label, softmax_vector in zip(qid_list, img_id_list, vqa_predictions, vqa_labels, vqa_scores_val)] ### 'softmax_vector': [float(i) for i in softmax_vector]

# print('total time taken for evaluating', ':', time.time()-start_time )

# start1=time.time()
# file_data['ques_ids'] = np.concatenate(all_qid,axis =0)
# file_data['ans_ids_predicted'] =  np.concatenate(all_gt_labels_random, axis=0)
# file_data['gt_ans_id_used'] = np.concatenate(all_ans_pred, axis=0)
# file_data['softmax_vector'] = np.concatenate(all_ss_vc, axis=0)
#
# print('total time taken for concatenating arrays ', ':', time.time()-start1)

start2=time.time()
with open(eval_file_pkl, 'wb') as f:
    pickle.dump(output_qids_answers, f, pickle.HIGHEST_PROTOCOL)             ## quick fix vedika  ## TypeError: Object of type 'ndarray' is not JSON serializable
#print('prediction file written to', eval_file_pkl)

print()

with open(os.path.join(
        result_dir, 'vqa_results_%s.txt' % cfg.TEST.SPLIT_VQA), 'w') as f:
    print('\nexp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
           accuracy, answer_correct, num_questions))
    print('exp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, cfg.TEST.ITER, cfg.TEST.SPLIT_VQA,
           accuracy, answer_correct, num_questions), file=f)

print()
print('results saved to:', eval_file_pkl)
# print('total time taken for dumping', cfg.TEST.SPLIT_VQA , ':', time.time()-start2 )

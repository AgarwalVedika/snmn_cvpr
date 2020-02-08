import argparse
import os
import json
import numpy as np
import tensorflow as tf
import ipdb
#to use this function: $python exp_vqa/to_validate.py --cfg exp_vqa/cfgs/{exp_name}.yaml
from models_vqa.model import Model
from models_vqa.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.vqa_train.data_reader import DataReader

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
assert cfg.EXP_NAME == os.path.basename(args.cfg).replace('.yaml', '')
if args.opts:
    merge_cfg_from_list(args.opts)

# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))


#
# # test options
# # --------------------------------------------------------------------------- #
# __C.TEST = AttrDict()
# __C.TEST.BATCH_SIZE = 64
# __C.TEST.USE_EMV = True
# __C.TEST.SPLIT_VQA =  'val2014'                          #original  'test-dev2015'                     ##############checking for vlaidation accuracies- visualize using tensorboard
# __C.TEST.SPLIT_LOC = 'REPLACE_THIS_WITH_GOOGLE_REF_TEST'
# __C.TEST.SNAPSHOT_FILE = './exp_vqa/tfmodel/%s/%08d'           #### load the snapshot you want to use
# __C.TEST.ITER = -1  # Needs to be supplied
#
# __C.TEST.RESULT_DIR = './exp_vqa/results/%s/%08d'
# __C.TEST.RESULT_DIR_val = './exp_vqa/results/%s'  ###########ADDDDDDD THIS TO CONFIG FILE!!!!!!
# __C.TEST.EVAL_FILE = './exp_vqa/eval_outputs/%s/vqa_OpenEnded_mscoco_%s_%s_%s_results.json'
# __C.TEST.VIS_SEPARATE_CORRECTNESS = False
# __C.TEST.NUM_VIS = 128
# __C.TEST.NUM_VIS_CORRECT = 100
# __C.TEST.NUM_VIS_INCORRECT = 100
# __C.TEST.VIS_DIR_PREFIX = 'vis'
# __C.TEST.STEPWISE_VIS = True  # Use the (new) stepwise visualization
# __C.TEST.VIS_SHOW_ANSWER = True
# __C.TEST.VIS_SHOW_STACK = True
# __C.TEST.VIS_SHOW_IMG = True
#
# __C.TEST.BBOX_IOU_THRESH = .5
#
#
# # --------------------------------------------------------------------------- #
# # --------------------------------------------------------------------------- #

#
# # Save snapshot                                              ##############	edit this: dont save the last checkpoint- save the one that gives good results-
# if ((
#         n_iter + 1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or  ######   so compare the last accurcay- if current greater- save the checkpoint-
#
#         (
#                 n_iter + 1) == cfg.TRAIN.MAX_ITER):  ####   next time when you save it again- delete the previous one- follow JJ's code
#     snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter + 1))
#     snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)


# Data files
imdb_file = cfg.IMDB_FILE % cfg.TEST.SPLIT_VQA
print(imdb_file)
data_reader = DataReader(
    imdb_file, shuffle=False, one_pass=True, batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE, load_gt_layout=True,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL)
num_vocab = data_reader.batch_loader.vocab_dict.num_vocab
num_choices = data_reader.batch_loader.answer_dict.num_vocab
module_names = data_reader.batch_loader.layout_dict.word_list

# Inputs and model
input_seq_batch = tf.placeholder(tf.int32, [None, None])
seq_length_batch = tf.placeholder(tf.int32, [None])
image_feat_batch = tf.placeholder(
    tf.float32, [None, cfg.MODEL.H_FEAT, cfg.MODEL.W_FEAT, cfg.MODEL.FEAT_DIM])
model = Model(
    input_seq_batch, seq_length_batch, image_feat_batch, num_vocab=num_vocab,
    num_choices=num_choices, module_names=module_names, is_training=False)

a = []
b = []
c = []
d = []

for TEST_ITER in np.arange(cfg.TRAIN.SNAPSHOT_INTERVAL , cfg.TRAIN.MAX_ITER + cfg.TRAIN.SNAPSHOT_INTERVAL , cfg.TRAIN.SNAPSHOT_INTERVAL):
	print(TEST_ITER)
	a.append(TEST_ITER)
	# Eval files
	if cfg.TEST.GEN_EVAL_FILE:
		eval_file = cfg.TEST.EVAL_FILE % (
			cfg.EXP_NAME, cfg.TEST.SPLIT_VQA, cfg.EXP_NAME, TEST_ITER)            ##########write a loop for test_iter??   _idea_vedika
		print('evaluation outputs will be saved to %s' % eval_file)
		os.makedirs(os.path.dirname(eval_file), exist_ok=True)
		answer_word_list = data_reader.batch_loader.answer_dict.word_list
		assert(answer_word_list[0] == '<unk>')
		output_qids_answers = []

	# Load snapshot
	if cfg.TEST.USE_EMV:
	    ema = tf.train.ExponentialMovingAverage(decay=0.9)  # decay doesn't matter
	    var_names = {
	        (ema.average_name(v) if v in model.params else v.op.name): v
	        for v in tf.global_variables()}
	else:
	    var_names = {v.op.name: v for v in tf.global_variables()}
	
	snapshot_file = cfg.TEST.SNAPSHOT_FILE % (cfg.EXP_NAME, TEST_ITER) ####they ar eloading the snapshot here- chose the one that does the best instead of the end one  ##comment_vedika
	print('snapshot file', snapshot_file)
	print(TEST_ITER)
	snapshot_saver = tf.train.Saver(var_names)
	snapshot_saver.restore(sess, snapshot_file)

	# Write results
	result_dir = cfg.TEST.RESULT_DIR % (cfg.EXP_NAME, TEST_ITER)
	print(result_dir)
	vis_dir = os.path.join(
	    result_dir, 'vqa_%s_%s' % (cfg.TEST.VIS_DIR_PREFIX, cfg.TEST.SPLIT_VQA))
	os.makedirs(result_dir, exist_ok=True)
	os.makedirs(vis_dir, exist_ok=True)

	# Run test
	answer_correct, num_questions = 0, 0
	for n_batch, batch in enumerate(data_reader.batches()):
	    fetch_list = [model.vqa_scores]
	    answer_incorrect = num_questions - answer_correct
	    fetch_list_val = sess.run(fetch_list, feed_dict={
	            input_seq_batch: batch['input_seq_batch'],
	            seq_length_batch: batch['seq_length_batch'],
	            image_feat_batch: batch['image_feat_batch']})
	
	#####edit_vedika lines 85-104
	    if cfg.TEST.VIS_SEPARATE_CORRECTNESS:
	        run_vis = (
	            answer_correct < cfg.TEST.NUM_VIS_CORRECT or
	            answer_incorrect < cfg.TEST.NUM_VIS_INCORRECT)
	    else:
	        run_vis = num_questions < cfg.TEST.NUM_VIS
	    if run_vis:
	        fetch_list.append(model.vis_outputs)
	    fetch_list_val = sess.run(fetch_list, feed_dict={
	            input_seq_batch: batch['input_seq_batch'],
	            seq_length_batch: batch['seq_length_batch'],
	            image_feat_batch: batch['image_feat_batch']})
	
	    # visualization
	
	    #def vis_batch_vqa(model, data_reader, batch, vis_outputs, start_idx,
	    #                  start_idx_correct, start_idx_incorrect, vis_dir):
	    #print('FETCH_LIST', fetch_list)
	    #ipdb.set_trace()
	    #print('FETCH LIST VAL- HAVE ATTENTION AND ALL', fetch_list_val)
	
	    run_vis = 0
	    if run_vis:
	        model.vis_batch_vqa(
	            data_reader, batch, fetch_list_val[-1], num_questions,                    #########print this fetch_list_val[-1]
	            answer_correct, answer_incorrect, vis_dir)
	
	    # compute accuracy
	    vqa_scores_val = fetch_list_val[0]
	    vqa_scores_val[:, 0] = -1e10  # remove <unk> answer
	    vqa_predictions = np.argmax(vqa_scores_val, axis=1)
	    if cfg.TEST.GEN_EVAL_FILE:
	        qid_list = batch['qid_list']
	        output_qids_answers += [
	            {'question_id': int(qid), 'answer': answer_word_list[p]}
	            for qid, p in zip(qid_list, vqa_predictions)]
	
	    if data_reader.batch_loader.load_answer:
	        vqa_labels = batch['answer_label_batch']
	    else:
	        # dummy labels with all -1 (so accuracy will be zero)
	        vqa_labels = -np.ones(vqa_scores_val.shape[0], np.int32)

		score_list = [np.sum(vqa_predictions[i] == vqa_labels[i]) for i in range(vqa_predictions.shape[0])]
		score_list2 = [1 if i > 2 else i / 3 for i in score_list]
		# answer_correct += np.sum(vqa_predictions == vqa_labels)  #answer_correct += np.sum(vqa_predictions == vqa_labels)  ORIGINALLY
		answer_correct += np.sum(score_list2)
		num_questions += len(vqa_labels)
		accuracy = answer_correct / num_questions

	    # answer_correct += np.sum(vqa_predictions == vqa_labels)
	    # num_questions += len(vqa_labels)
	    # accuracy = answer_correct / num_questions

	    if n_batch % 20 == 0:
	        print('exp: %s, iter = %d, accumulated accuracy on %s = %f (%d / %d)' %
	              (cfg.EXP_NAME, TEST_ITER, cfg.TEST.SPLIT_VQA,
	               accuracy, answer_correct, num_questions))
	
	with open(eval_file, 'w') as f:
	        json.dump(output_qids_answers, f, indent=2)
	        print('prediction file written to', eval_file)
	
	with open(os.path.join(
	        result_dir, 'vqa_results_%s.txt' % cfg.TEST.SPLIT_VQA), 'w') as f:
	    print('\nexp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
	          (cfg.EXP_NAME, TEST_ITER, cfg.TEST.SPLIT_VQA,
	           accuracy, answer_correct, num_questions))
	    print('exp: %s, iter = %d, final accuracy on %s = %f (%d / %d)' %
	          (cfg.EXP_NAME, TEST_ITER, cfg.TEST.SPLIT_VQA,
	           accuracy, answer_correct, num_questions), file=f)
	b.append(accuracy)
	c.append(answer_correct)
	d.append(num_questions)
	
print('test_iter', a)
print('final accuracy', b)
print('answer_correct', c)
print('num_questions', d)


val_result_dir = cfg.TEST.RESULT_DIR_val % (cfg.EXP_NAME)
with open(os.path.join(
	        val_result_dir, 'val_results_%s.txt' % cfg.TEST.SPLIT_VQA), 'w') as f:   
	print('prediction file written to', val_result_dir, 'val_results_%s.txt' % cfg.TEST.SPLIT_VQA)     
	f.write('test_iter:'+ str(a))
	f.write('final accuracy:'+ str(b))
	f.write('ans_correct:'+ str(c))
	f.write('num_questions:'+ str(d))



import argparse
import os
import json
import numpy as np
import tensorflow as tf
import ipdb
import pickle
import time
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

start=time.time()

# Data files
imdb_file = cfg.IMDB_FILE % cfg.VAL.SPLIT_VQA
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

# Load snapshot
if cfg.TEST.USE_EMV:
    ema = tf.train.ExponentialMovingAverage(decay=0.9)  # decay doesn't matter
    var_names = {
        (ema.average_name(v) if v in model.params else v.op.name): v
        for v in tf.global_variables()}
else:
    var_names = {v.op.name: v for v in tf.global_variables()}

abcd = {}
file_dict = {}

val_epoch_acc = []
for EPOCH in range(cfg.TRAIN.MAX_EPOCH):#(cfg.TRAIN.MAX_EPOCH):
    snapshot_file = cfg.TEST.SNAPSHOT_FILE % (cfg.EXP_NAME, EPOCH+1)    # SNAPSHOT_FILE: './exp_vqa/tfmodel/%s/%04d'
    snapshot_saver = tf.train.Saver(var_names)
    snapshot_saver.restore(sess, snapshot_file)

    # Run test
    answer_correct, num_questions = 0, 0
    for n_batch, batch in enumerate(data_reader.batches()):
        #print(n_batch)
        fetch_list = [model.vqa_scores]
        answer_incorrect = num_questions - answer_correct
        fetch_list_val = sess.run(fetch_list, feed_dict={
                input_seq_batch: batch['input_seq_batch'],
                seq_length_batch: batch['seq_length_batch'],
                image_feat_batch: batch['image_feat_batch']})

        # compute accuracy
        vqa_scores_val = fetch_list_val[0]
        vqa_scores_val[:, 0] = -1e10  # remove <unk> answer
        vqa_predictions = np.argmax(vqa_scores_val, axis=1)

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

    print('exp: %s, EPOCH = %d, accuracy on %s = %f (%d / %d)' %
          (cfg.EXP_NAME, EPOCH+1, cfg.VAL.SPLIT_VQA,accuracy, answer_correct, num_questions))
    val_epoch_acc.append(accuracy)
    file_dict[EPOCH+1]=accuracy

abcd['epoch_accuracy']= file_dict


print(val_epoch_acc)
print('total time taken for validating models:',  time.time()-start)

saving_dir = os.path.join('/BS/vedika2/nobackup/snmn/vedi_logs',  cfg.EXP_NAME)
os.makedirs(saving_dir, exist_ok=True)
saving_file = cfg.VAL.SPLIT_VQA + '.json'

with open(os.path.join(saving_dir, saving_file), 'w') as f:
    json.dump(abcd,f)

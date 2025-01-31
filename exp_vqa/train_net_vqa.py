import argparse
import os
import numpy as np
import tensorflow as tf

from models_vqa.model import Model
from models_vqa.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.vqa_train.data_reader import DataReader
import ipdb
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', required=True)
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
merge_cfg_from_file(args.cfg)
assert cfg.EXP_NAME == os.path.basename(args.cfg).replace('.yaml', '')
if args.opts:
    merge_cfg_from_list(args.opts)

start = time.time()

#print('123')
# Start session
os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth=cfg.GPU_MEM_GROWTH)))

# Data files
imdb_file = cfg.IMDB_FILE % cfg.TRAIN.SPLIT_VQA                  ####__C.TRAIN.SPLIT_VQA = 'trainval2014'ipdb.set_trace()
len_train_set = len(np.load(imdb_file))
data_reader = DataReader(
    imdb_file, shuffle=True, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,
    vocab_question_file=cfg.VOCAB_QUESTION_FILE, T_encoder=cfg.MODEL.T_ENCODER,
    vocab_answer_file=cfg.VOCAB_ANSWER_FILE,
    load_gt_layout=cfg.TRAIN.USE_GT_LAYOUT,
    vocab_layout_file=cfg.VOCAB_LAYOUT_FILE, T_decoder=cfg.MODEL.T_CTRL,
    load_soft_score=cfg.TRAIN.VQA_USE_SOFT_SCORE)

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
    num_choices=num_choices, module_names=module_names, is_training=True)

# Loss function
if cfg.TRAIN.VQA_USE_SOFT_SCORE:
    soft_score_batch = tf.placeholder(tf.float32, [None, num_choices])
    # Summing, instead of averaging over the choices
    loss_vqa = float(num_choices) * tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            logits=model.vqa_scores, labels=soft_score_batch))
else:
    print()
    print('you need to modify to get answer_label_batch as you did for test/val case')
    ipdb.set_trace()
    answer_label_batch = tf.placeholder(tf.int32, [None])
    loss_vqa = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model.vqa_scores, labels=answer_label_batch))
if cfg.TRAIN.USE_GT_LAYOUT:
    gt_layout_batch = tf.placeholder(tf.int32, [None, None])
    loss_layout = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=model.module_logits, labels=gt_layout_batch))
else:
    loss_layout = tf.convert_to_tensor(0.)
loss_rec = model.rec_loss
loss_train = (loss_vqa * cfg.TRAIN.VQA_LOSS_WEIGHT +
              loss_layout * cfg.TRAIN.LAYOUT_LOSS_WEIGHT +
              loss_rec * cfg.TRAIN.REC_LOSS_WEIGHT)
loss_total = loss_train + cfg.TRAIN.WEIGHT_DECAY * model.l2_reg


#ipdb.set_trace()
# Train with Adam
solver = tf.train.AdamOptimizer(learning_rate=cfg.TRAIN.SOLVER.LR)
# solver_op = solver.minimize(loss_total)

if cfg.TRAIN.CLIP_GRADIENTS:
    grads_and_vars = solver.compute_gradients(loss_total)
    if cfg.TRAIN.CLIP_GRADIENTS:
        print('clipping gradients to max norm: %f' % cfg.TRAIN.GRAD_MAX_NORM)
        gradients, variables = zip(*grads_and_vars)
        gradients, _ = tf.clip_by_global_norm(gradients, cfg.TRAIN.GRAD_MAX_NORM)
        grads_and_vars = zip(gradients, variables)
    solver_op = solver.apply_gradients(grads_and_vars)
else:
    solver_op = solver.minimize(loss_total)


# Save moving average of parameters
#changes: lines 92 to 96: edit_vedika : not influencing the lr- keep it same htorouhout
ema = tf.train.ExponentialMovingAverage(decay=cfg.TRAIN.EMV_DECAY)
ema_op = ema.apply(model.params)
with tf.control_dependencies([solver_op]):
    train_op = tf.group(ema_op)# some decay    ##TODO
#train_op = solver_op

# Save snapshot
snapshot_dir = cfg.TRAIN.SNAPSHOT_DIR % cfg.EXP_NAME
os.makedirs(snapshot_dir, exist_ok=True)
snapshot_saver = tf.train.Saver(max_to_keep=None)  # keep all snapshots
if cfg.TRAIN.START_ITER > 0:
    snapshot_file = os.path.join(snapshot_dir, "%08d" % cfg.TRAIN.START_ITER)
    print('resume training from %s' % snapshot_file)
    snapshot_saver.restore(sess, snapshot_file)
else:
    sess.run(tf.global_variables_initializer())
    if cfg.TRAIN.INIT_FROM_WEIGHTS:
        snapshot_saver.restore(sess, cfg.TRAIN.INIT_WEIGHTS_FILE)
        print('initialized from %s' % cfg.TRAIN.INIT_WEIGHTS_FILE)
# Save config
np.save(os.path.join(snapshot_dir, 'cfg.npy'), np.array(cfg))

# Write summary to TensorBoard
log_dir = cfg.TRAIN.LOG_DIR % cfg.EXP_NAME
os.makedirs(log_dir, exist_ok=True)
log_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())                      ####originally they had jsut for val- need two summary_writers for train and val
#log_writer_train = tf.summary.FileWriter(log_dir + '/train', tf.get_default_graph())
#log_writer_val = tf.summary.FileWriter(log_dir + '/val' , tf.get_default_graph())

loss_vqa_ph = tf.placeholder(tf.float32, [])
loss_layout_ph = tf.placeholder(tf.float32, [])
loss_rec_ph = tf.placeholder(tf.float32, [])
accuracy_ph = tf.placeholder(tf.float32, [])
avg_accuracy_ph = tf.placeholder(tf.float32, [])
summary_trn = []
summary_trn.append(tf.summary.scalar("loss/vqa", loss_vqa_ph))
summary_trn.append(tf.summary.scalar("loss/layout", loss_layout_ph))
summary_trn.append(tf.summary.scalar("loss/rec", loss_rec_ph))
summary_trn.append(tf.summary.scalar("eval/vqa/accuracy", accuracy_ph))
summary_trn.append(tf.summary.scalar("eval/vqa/accuracy_avg", avg_accuracy_ph))
log_step_trn = tf.summary.merge(summary_trn)

# Run training
avg_accuracy, accuracy_decay = 0., 0.99
for n_batch, batch in enumerate(data_reader.batches()):
    n_iter = n_batch + cfg.TRAIN.START_ITER
    n_epoch = (n_iter*cfg.TRAIN.BATCH_SIZE)/len_train_set   ### SO FOR TRAIN SET; 3466 ITERATIONS WOULD MEAN ONE EPOCH
    if n_iter >= cfg.TRAIN.MAX_ITER:
        break

    feed_dict = {input_seq_batch: batch['input_seq_batch'],
                 seq_length_batch: batch['seq_length_batch'],
                 image_feat_batch: batch['image_feat_batch']}
    if cfg.TRAIN.VQA_USE_SOFT_SCORE:
        feed_dict[soft_score_batch] = batch['soft_score_batch']
    else:
        feed_dict[answer_label_batch] = batch['answer_label_batch']
    if cfg.TRAIN.USE_GT_LAYOUT:
        feed_dict[gt_layout_batch] = batch['gt_layout_batch']

    vqa_scores_val, loss_vqa_val, loss_layout_val, loss_rec_val, _ = sess.run(
        (model.vqa_scores, loss_vqa, loss_layout, loss_rec, train_op),
        feed_dict)

    # compute accuracy
    vqa_labels = batch['answer_label_batch']
    vqa_predictions = np.argmax(vqa_scores_val, axis=1)
    accuracy = np.mean(vqa_predictions == vqa_labels)
    avg_accuracy += (1-accuracy_decay) * (accuracy-avg_accuracy)


    # Add to TensorBoard summary																##########here they write the validation loss/accuracies to the summary_writer
    if (n_iter+1) % cfg.TRAIN.LOG_INTERVAL == 0:
        print("exp: %s, iter = %d\n\t" % (cfg.EXP_NAME, n_iter+1) +
              "loss (vqa) = %f, loss (layout) = %f, loss (rec) = %f\n\t" % (
                loss_vqa_val, loss_layout_val, loss_rec_val) +
              "accuracy (cur) = %f, accuracy (avg) = %f" % (
                accuracy, avg_accuracy))
        summary = sess.run(log_step_trn, {
            loss_vqa_ph: loss_vqa_val,
            loss_layout_ph: loss_layout_val,
            loss_rec_ph: loss_rec_val,
            accuracy_ph: accuracy,
            avg_accuracy_ph: avg_accuracy})
        log_writer.add_summary(summary, n_iter+1)


    # Save snapshot                                              ##############	edit this: dont save the last checkpoint- save the one that gives good results-
    if ((n_iter+1) % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or						######   so compare the last accurcay- if current greater- save the checkpoint-

            (n_iter+1) == cfg.TRAIN.MAX_ITER):									  ####   next time when you save it again- delete the previous one- follow JJ's code
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (n_iter+1))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        #  if #d[n_iter+1]> d[n_iter] or-- somehting like this
        #snapshot_file_del = os.path.join(snapshot_dir, "%08d" % (n_iter)
        #del snapshot_file_del            ##############use tf.keras-save model checkpoint and tf.keras.fit---- to handle val/train splits
        #print('snapshot saved to ' + snapshot_file)

print()
print('total time taken to train the model for', cfg.TRAIN.MAX_ITER, 'iterations:', time.time()-start)
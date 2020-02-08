import argparse
import os
import numpy as np
import tensorflow as tf
import pickle
from models_vqa.model import Model
from models_vqa.config import (
    cfg, merge_cfg_from_file, merge_cfg_from_list)
from util.vqa_train.data_reader import DataReader
import ipdb
import json
import time
import math


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
len_imd = len(np.load(imdb_file))

data_reader = DataReader(
    imdb_file, shuffle=cfg.TRAIN.SHUFFLE, one_pass=False, batch_size=cfg.TRAIN.BATCH_SIZE,  # edit_vedika one_pass= True
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
if cfg.TRAIN.START_EPOCH> 0:
    snapshot_file = os.path.join(snapshot_dir, "%08d" % cfg.TRAIN.START_EPOCH)
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


# checking how many iterations make an epoch
num_iter_epoch = math.ceil((len_imd)/cfg.TRAIN.BATCH_SIZE)    ### SO FOR only TRAIN SET; 3466.85= 3467 ITERATIONS WOULD MEAN ONE EPOCH
abcd = {}
# Run training
avg_accuracy, accuracy_decay = 0., 0.99
answer_correct, num_questions = 0, 0
train_epoch_acc = []
file_dict = {}
for n_batch, batch in enumerate(data_reader.batches()):
    n_iter = n_batch + cfg.TRAIN.START_EPOCH*num_iter_epoch  #cfg.TRAIN.START_ITER
    N_EPOCH = (n_iter+1)/num_iter_epoch
    #assert math.ceil(N_EPOCH)==cfg.TRAIN.START_EPOCH+1
    #ipdb.set_trace()
    #print('current epoch:', ((n_iter+1)/num_iter_epoch))
    #n_epoch = (n_iter*cfg.TRAIN.BATCH_SIZE)/len_train_set   ### SO FOR only TRAIN SET; 3466.85= 3467 ITERATIONS WOULD MEAN ONE EPOCH
    if n_iter/num_iter_epoch >= cfg.TRAIN.MAX_EPOCH:
        break

    feed_dict = {input_seq_batch: batch['input_seq_batch'],
                 seq_length_batch: batch['seq_length_batch'],
                 image_feat_batch: batch['image_feat_batch']}
    if cfg.TRAIN.VQA_USE_SOFT_SCORE:
        feed_dict[soft_score_batch] = batch['soft_score_batch']
    else:
        print()
        print('you need to modify to get answer_label_batch as you did for test/val case')
        ipdb.set_trace()
        feed_dict[answer_label_batch] = batch['answer_label_batch']
    if cfg.TRAIN.USE_GT_LAYOUT:
        feed_dict[gt_layout_batch] = batch['gt_layout_batch']
    
    vqa_scores_val, loss_vqa_val, loss_layout_val, loss_rec_val, _ = sess.run(
        (model.vqa_scores, loss_vqa, loss_layout, loss_rec, train_op),
        feed_dict)

    # compute accuracy
    vqa_labels = batch['answer_label_batch']
    vqa_predictions = np.argmax(vqa_scores_val, axis=1)

    #accuracy_epoch.append(accuracy)

    score_list =  [np.sum(vqa_predictions[i] == vqa_labels[i]) for i in range(vqa_predictions.shape[0])]
    score_list2 = [1 if i > 2 else i / 3 for i in score_list]
    #answer_correct += np.sum(vqa_predictions == vqa_labels)  #answer_correct += np.sum(vqa_predictions == vqa_labels)  ORIGINALLY
    answer_correct += np.sum(score_list2)
    num_questions += len(vqa_labels)
    accuracy = answer_correct / num_questions

    # Save snapshot                                              ##############	edit this: dont save the last checkpoint- save the one that gives good results- 
    if (n_iter+1) % num_iter_epoch == 0 :
        print('epoch = %d,  accuracy = %f (%d / %d)' %
              (N_EPOCH, accuracy, answer_correct, num_questions))
        train_epoch_acc.append(accuracy)
        file_dict[N_EPOCH] = accuracy
        answer_correct, num_questions = 0, 0
        snapshot_file = os.path.join(snapshot_dir, "%08d" % (N_EPOCH))
        snapshot_saver.save(sess, snapshot_file, write_meta_graph=False)
        #  if #d[n_iter+1]> d[n_iter] or-- somehting like this
        #snapshot_file_del = os.path.join(snapshot_dir, "%08d" % (n_iter)
        #del snapshot_file_del            ##############use tf.keras-save model checkpoint and tf.keras.fit---- to handle val/train splits
        #print('snapshot saved to ' + snapshot_file)


abcd['learning rate']= cfg.TRAIN.SOLVER.LR
abcd['epoch_accuracy']= file_dict

saving_dir = os.path.join('/BS/vedika2/nobackup/snmn/vedi_logs',  cfg.EXP_NAME)
os.makedirs(saving_dir, exist_ok=True)
saving_file = cfg.TRAIN.SPLIT_VQA + '.json'
with open(os.path.join(saving_dir, saving_file), 'w') as f:
    json.dump(abcd,f)


print('accuracy at every epoch:', train_epoch_acc)
print('number of epoch the model is trained for:', (n_iter+1/num_iter_epoch) )
print('total time taken to train the model for', cfg.TRAIN.MAX_ITER, 'iterations:', time.time()-start)


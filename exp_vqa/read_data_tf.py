import tensorflow as tf
import ipdb
import numpy as np


#for summary in tf.train.summary_iterator("/BS/vedika2/work/snmn/exp_vqa/tb/vqa_v1_gt_layout_3/events.out.tfevents.1544551694.d2volta17"):
#    # Perform custom processing in here.



from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
#event_acc = EventAccumulator('/BS/vedika2/work/snmn/exp_vqa/tb/vqa_v1_scratch_3/')
event_acc = EventAccumulator('/BS/vedika2/work/snmn/exp_vqa/tb/vqa_v1_scratch/')               ####### path
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('eval/vqa/accuracy'))
#ipdb.set_trace()
#for i in np.arange(2500,202500,2500):
#    print(step_nums[i], vals[i])

#ipdb.set_trace()
w_times = np.array(w_times)
step_nums = np.array(step_nums)
vals = np.array(vals)

new_val = []
step = []
for i in range(len(vals)):
    if step_nums[i] % 50 == 0:                            # divide by  train.snapshot.interval
        #if (i)%125 ==0:
        new_val.append(vals[i])
        step.append(step_nums[i])
print(new_val)
print(step)
#import ipdb
#ipdb.set_trace()

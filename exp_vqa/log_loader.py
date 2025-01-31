import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt
import ipdb

def plot_tensorflow_log(path):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())
    ipdb.set_trace()

    training_accuracies =   event_acc.Scalars('eval/vqa/accuracy')
    validation_accuracies = event_acc.Scalars('loss/vqa')
    print(training_accuracies)
    print(validation_accuracies)
    steps = 10
    x = np.arange(steps)
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]

    plt.plot(x, y[:,0], label='training accuracy')
    plt.plot(x, y[:,1], label='loss/vqa')

    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    plt.show()


if __name__ == '__main__':
    log_file = "/BS/vedika2/work/snmn/exp_vqa/tb/vqa_v1_scratch/events.out.tfevents.1542190877.d2volta18"
plot_tensorflow_log(log_file)
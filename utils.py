import os
import time

import numpy as np

import keras

def find_key_by_str(keys, needle):
    for key in keys:
        if needle in key:
            return key
    raise ValueError("%s not found in keys" % (needle))

class LandcoverResults(keras.callbacks.Callback):

    def __init__(self, log_dir=None, time_budget=None, verbose=False, model=None):

        self.scores = {
            "mb_train_accuracy":[],
            "mb_train_crossentropy":[],
            "mb_train_jaccard":[],
            "epoch_train_accuracy":[],
            "epoch_train_crossentropy":[],
            "epoch_train_jaccard":[],
            "epoch_val_accuracy":[],
            "epoch_val_crossentropy":[],
            "epoch_val_jaccard":[]
        }
        
        self.epoch_times = []
        self.mb_times = []
        self.verbose = verbose
        self.time_budget = time_budget
        self.log_dir = log_dir

        self.batch_num = 0
        self.epoch_num = 0

        self.model_inst = model

        if self.log_dir is not None:
            self.train_mb_fn = os.path.join(log_dir, "train_minibatch_history.txt")
            self.train_epoch_fn = os.path.join(log_dir, "train_epoch_history.txt")
            self.val_epoch_fn = os.path.join(log_dir, "val_epoch_history.txt")
            f = open(self.train_mb_fn,"w")
            f.write("Batch Number,Time Elapsed,Train Crossentropy,Train Accuracy,Train Jaccard\n")
            f.close()
            f = open(self.train_epoch_fn,"w")
            f.write("Epoch Number,Time Elapsed,Train Crossentropy,Train Accuracy,Train Jaccard\n")
            f.close()
            f = open(self.val_epoch_fn,"w")
            f.write("Epoch Number,Time Elapsed,Val Crossentropy,Val Accuracy,Train Jaccard\n")
            f.close()

    def on_train_begin(self, logs={}):
        self.train_start_time = time.time()
        
    def on_batch_begin(self, batch, logs={}):
        self.mb_start_time = float(time.time())

    def on_batch_end(self, batch, logs={}):
        #print("batch",logs)
        t = time.time() - self.mb_start_time

        jaccard_key = find_key_by_str(logs.keys(), "jaccard_loss")
        crossentropy_key = "categorical_crossentropy" if "categorical_crossentropy" in logs.keys() else "loss"
        accuracy_key = "accuracy" if "accuracy" in logs.keys() else "acc"

        self.scores["mb_train_accuracy"].append(logs[accuracy_key])
        self.scores["mb_train_crossentropy"].append(logs[crossentropy_key])
        self.scores["mb_train_jaccard"].append(logs[jaccard_key])

        if self.log_dir is not None:
            f = open(self.train_mb_fn,"a")
            f.write("%d,%f,%f,%f,%f\n" % (self.batch_num, t, logs[crossentropy_key], logs[accuracy_key], logs[jaccard_key]))
            f.close()

        self.mb_times.append(t)
        self.batch_num += 1
        
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start_time = float(time.time())
    
    def on_epoch_end(self, epoch, logs=None):
        #print("epoch",logs)
        t = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time

        if self.time_budget is not None:
            if total_time >= self.time_budget:
                try:
                    self.model.stop_training = True
                except Exception:
                    pass
                try:
                    self.model_inst.stop_training = True
                except Exception:
                    pass

        jaccard_key = find_key_by_str(logs.keys(), "jaccard_loss")
        crossentropy_key = "categorical_crossentropy" if "categorical_crossentropy" in logs.keys() else "loss"
        accuracy_key = "accuracy" if "accuracy" in logs.keys() else "acc"

        val_jaccard_key = find_key_by_str(logs.keys(), "val_jaccard_loss")
        val_crossentropy_key = "val_categorical_crossentropy" if "val_categorical_crossentropy" in logs.keys() else "val_loss"
        val_accuracy_key = "val_accuracy" if "val_accuracy" in logs.keys() else "val_acc"
        
        self.scores["epoch_val_accuracy"].append(logs[val_accuracy_key])
        self.scores["epoch_val_crossentropy"].append(logs[val_crossentropy_key])
        self.scores["epoch_val_jaccard"].append(logs[val_jaccard_key])

        self.scores["epoch_train_accuracy"].append(logs[accuracy_key])
        self.scores["epoch_train_crossentropy"].append(logs[crossentropy_key])
        self.scores["epoch_train_jaccard"].append(logs[jaccard_key])
        
        if self.log_dir is not None:
            f = open(self.train_epoch_fn,"a")
            f.write("%d,%f,%f,%f,%f\n" % (self.epoch_num, t, logs[crossentropy_key], logs[accuracy_key], logs[jaccard_key]))
            f.close()
            f = open(self.val_epoch_fn,"a")
            f.write("%d,%f,%f,%f,%f\n" % (self.epoch_num, t, logs[val_crossentropy_key], logs[val_accuracy_key], logs[val_jaccard_key]))
            f.close()

        if self.verbose:
            print("")
            print("Val ACC: %f, Val CROSSENTROPY: %f, VAL JACCARD: %f" % (logs[val_accuracy_key], logs[val_crossentropy_key], logs[val_jaccard_key]))
            print("")

        self.epoch_times.append(t)
        self.epoch_num += 1
#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
# pylint: skip-file
#
# Copyright © 2017 Caleb Robinson <calebrob6@gmail.com>
#
# Distributed under terms of the MIT license.
'''Runs minibatch sampling algorithms on migration datasets
'''
import sys
import os

# Here we look through the args to find which GPU we should use
# We must do this before importing keras, which is super hacky
# See: https://stackoverflow.com/questions/40690598/can-keras-with-tensorflow-backend-be-forced-to-use-cpu-or-gpu-at-will
# TODO: This _really_ should be part of the normal argparse code.
def parse_args(args, key):
    def is_int(s):
        try: 
            int(s)
            return True
        except ValueError:
            return False
    for i, arg in enumerate(args):
        if arg == key:
            if i+1 < len(sys.argv):
                if is_int(args[i+1]):
                    return args[i+1]
    return None
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GPU_ID = parse_args(sys.argv, "--gpu")
if GPU_ID is not None: # if we passed `--gpu INT`, then set the flag, else don't
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import shutil
import time
import argparse
import datetime

import numpy as np

import utils
import models
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def data_generator(fns, batch_size, input_size, output_size, num_channels, num_classes, verbose=None):
    file_indices = list(range(len(fns)))
    
    x_batch = np.zeros((batch_size, input_size, input_size, num_channels), dtype=np.float32)
    y_batch = np.zeros((batch_size, output_size, output_size, num_classes), dtype=np.float32)
    
    counter = 0
    while 1:
        np.random.shuffle(file_indices)
        
        for i in range(0, len(file_indices), batch_size):

            if i + batch_size >= len(file_indices): # if we don't have enough samples left, just quit and reshuffle
                break

            batch_idx = 0
            for j in range(i, i+batch_size):
                data = np.load(fns[file_indices[j]]).squeeze()
                data = np.rollaxis(data, 0, 3)

                x_batch[batch_idx] = data[:,:,:4]
                y_batch[batch_idx] = to_categorical(data[:,:,4], num_classes=num_classes)
                batch_idx += 1

            yield (x_batch.copy(), y_batch.copy())
            if verbose is not None:
                print("%s yielded %d" % (verbose, counter))
            counter += 1

def do_args(arg_list, name):
    parser = argparse.ArgumentParser(description=name)

    parser.add_argument("-v", "--verbose", action="store", dest="verbose", type=int, help="Verbosity of keras.fit", default=2)
    parser.add_argument("--output", action="store", dest="output", type=str, help="Output base directory", required=True)
    parser.add_argument("--name", action="store", dest="name", type=str, help="Experiment name", required=True)
    parser.add_argument("--gpu", action="store", dest="gpu", type=int, help="GPU id to use", required=False)

    parser.add_argument("--training_patches", action="store", dest="training_patches_fn", type=str, help="Path to file containing training patches", required=True)
    parser.add_argument("--model_type", action="store", dest="model_type", type=str, \
        choices=["baseline", "extended", "extended_bn", "extended2_bn", "unet1", "unet2", "unet3"], \
        help="Model architecture to use", required=True
    )

    # training arguments
    parser.add_argument("--batch_size", action="store", type=eval, help="Batch size", default="128")
    parser.add_argument("--time_budget", action="store", type=int, help="Time limit", default=3600*3)
    parser.add_argument("--learning_rate", action="store", type=float, help="Learning rate", default=0.003)
    parser.add_argument("--loss", action="store", type=str, help="Loss function", choices=["crossentropy", "jaccard"], default=None)

    return parser.parse_args(arg_list)


def main():
    prog_name = sys.argv[0]
    args = do_args(sys.argv[1:], prog_name)

    verbose = args.verbose
    output = args.output
    name = args.name

    training_patches_fn = args.training_patches_fn

    log_dir = os.path.join(output, name)

    assert os.path.exists(log_dir), "Output directory doesn't exist"

    f = open(os.path.join(log_dir, "args.txt"), "w")
    f.write("%s\n"  % (" ".join(sys.argv)))
    f.close()

    print("Starting %s at %s" % (prog_name, str(datetime.datetime.now())))
    start_time = float(time.time())

    #------------------------------
    # Step 1, load data
    #------------------------------

    f = open(training_patches_fn, "r")
    training_patches = f.read().strip().split("\n")
    f.close()

    highres_patches = []
    for fn in training_patches:
        parts = fn.split("-")
        parts = np.array(list(map(int, parts[2].split("_"))))
        if parts[0] == 0:
            highres_patches.append(fn)


    #------------------------------
    # Step 2, run experiment
    #------------------------------

    model_type = args.model_type
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    time_budget = args.time_budget
    loss = args.loss
    training_steps_per_epoch = len(highres_patches) // batch_size // 16
    validation_steps_per_epoch = 40

    #training_steps_per_epoch = 10
    #validation_steps_per_epoch = 2
    print("Number of training steps per epoch: %d" % (training_steps_per_epoch))


    # Build the model
    if model_type == "baseline":
        model = models.baseline_model_landcover((240,240,4), 7, lr=learning_rate, loss=loss)
    elif model_type == "extended":
        model = models.extended_model_landcover((240,240,4), 7, lr=learning_rate, loss=loss)
    elif model_type == "extended_bn":
        model = models.extended_model_bn_landcover((240,240,4), 7, lr=learning_rate, loss=loss)
    elif model_type == "extended2_bn":
        model = models.extended2_model_bn_landcover((240,240,4), 7, lr=learning_rate, loss=loss)
    elif model_type == "unet1":
        model = models.unet_landcover(
            (240,240,4), out_ch=7, start_ch=64, depth=3, inc_rate=2., activation='relu', 
            dropout=0.5, batchnorm=True, maxpool=True, upconv=True, residual=False, lr=learning_rate, loss=loss
        )
    elif model_type == "unet2":
        model = models.unet_landcover(
            (240,240,4), out_ch=7, start_ch=32, depth=4, inc_rate=2., activation='relu', 
            dropout=False, batchnorm=True, maxpool=True, upconv=True, residual=False, lr=learning_rate, loss=loss
        )
    model.summary()
    

    def schedule_decay(epoch, lr, decay=0.001):
        if epoch>=10:
            lr = lr * 1/(1 + decay * epoch)
        return lr
    
    def schedule_stepped(epoch, lr):
        if epoch < 10:
            return 0.003
        elif epoch < 20:
            return 0.0003
        elif epoch < 30:
            return 0.00015
        else:
            return 0.00003

    validation_callback = utils.LandcoverResults(log_dir=log_dir, time_budget=time_budget, verbose=False)
    learning_rate_callback = LearningRateScheduler(schedule_stepped, verbose=1)
    model_checkpoint_callback = ModelCheckpoint(
        os.path.join(log_dir, "model_{epoch:02d}.h5"),
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        period=1
    )

    model.fit_generator(
        data_generator(highres_patches, batch_size, 240, 240, 4, 7),
        steps_per_epoch=training_steps_per_epoch,
        epochs=10**6,
        verbose=1,
        validation_data=data_generator(highres_patches, batch_size, 240, 240, 4, 7),
        validation_steps=validation_steps_per_epoch,
        max_queue_size=64,
        workers=1,
        use_multiprocessing=True,
        callbacks=[validation_callback, learning_rate_callback, model_checkpoint_callback],
        initial_epoch=0 
    )

    #------------------------------
    # Step 3, save models
    #------------------------------
    model.save(os.path.join(log_dir, "final_model.h5"))

    model_json = model.to_json()
    with open(os.path.join(log_dir,"final_model.json"), "w") as json_file:
        json_file.write(model_json)
    model.save_weights(os.path.join(log_dir, "final_model_weights.h5"))

    print("Finished in %0.4f seconds" % (time.time() - start_time))

if __name__ == "__main__":
    main()
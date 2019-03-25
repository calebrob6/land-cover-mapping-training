import argparse
import json, shutil, os
import torch
from pytorch.utils.experiment_utils import (
    CheckpointSaver, improve_reproducibility, set_all_rngs, NamespaceFromDict
)


"""
@uthor: Anthony Ortiz
Date: 03/25/2019
Last Modified: 03/25/2019
"""
def parse_int_list(s):
    if s == '': return ()
    return tuple(int(n) for n in s.split(','))

parser = argparse.ArgumentParser()

### JSON FILE IF NEW EXPERIMENT
parser.add_argument('--config-file', type=str,
                    help="json file containing the configuration")
# CHECKPOINT FILE IF RESUMING
parser.add_argument('--checkpoint-file', type=str,
                    help="file to resume training from")

### FILE I/O SETTINGS
parser.add_argument('--data-root', type=str, required=True,
                    help="root dir of the data set")
parser.add_argument('--train-dir', type=str, required=True,
                    help="path where checkpoints are saved")
parser.add_argument('--backup-dir', type=str, required=True,
                    help='backup dir for long-term storage of parameters')

### GENERAL EXPERIMENT SETTINGS
parser.add_argument('--debug', action='store_true', help=(
    "whether to log debug information, such as norms of weights and gradients."
))

args = parser.parse_args()

assert (args.checkpoint_file is None) ^ (args.config_file is None), (
    "either specify a config-file for a new experiment or a checkpoint-file "
    "for resuming an existing experiment")

if args.config_file is not None:
    resume_run = False

    # load config file
    config_dict = json.load(open(args.config_file))
    config = NamespaceFromDict(config_dict)

    args_dict = vars(args)
    # save command-line arguments to train and backup dir
    json.dump(args_dict, open(os.path.join(args.train_dir, 'args.json'), 'w'),
            indent=4, sort_keys=True)
    json.dump(args_dict, open(os.path.join(args.backup_dir, 'args.json'), 'w'),
            indent=4, sort_keys=True)
    shutil.copy(args.config_file, os.path.join(args.train_dir, 'config.json'))
    shutil.copy(args.config_file, os.path.join(args.backup_dir, 'config.json'))

else:
    resume_run = True

    # load checkpoint and corresponding args.json overwriting command-line args
    checkpoint_file = args.checkpoint_file  # backup checkpoint file path
    args_dict = json.load(open(os.path.join(args.train_dir, 'args.json')))
    config_dict = json.load(open(os.path.join(args.train_dir, 'config.json')))
    args = NamespaceFromDict(args_dict)
    config = NamespaceFromDict(config_dict)


print(f'using training directory {args.train_dir}')

# improve reproducibility
improve_reproducibility(seed=config.rand_seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

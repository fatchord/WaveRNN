import numpy as np
import os
import re
import json
import datetime
import torch
import glob
import shutil
import subprocess


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""

    checkpoint_files = glob.glob(experiment_path + "/**/*.pth.tar", recursive=True)
    if len(checkpoint_files) < 1:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    # try:
    #     subprocess.check_output(['git', 'diff-index', '--quiet',
    #                              'HEAD'])  # Verify client is clean
    # except:
    #     raise RuntimeError(
    #         " !! Commit before training to get the commit hash.")
    commit = subprocess.check_output(['git', 'rev-parse', '--short',
                                      'HEAD']).decode().strip()
    print(' > Git Hash: {}'.format(commit))
    return commit


def create_experiment_folder(root_path, model_name):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    # if debug:
        # commit_hash = 'debug'
    # else:
    commit_hash = get_commit_hash()
    output_folder = os.path.join(
        root_path, model_name + '-' + date_str + '-' + commit_hash)
    os.makedirs(output_folder, exist_ok=True)
    print(" > Experiment folder: {}".format(output_folder))
    return output_folder


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class SlowStartLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super(SlowStartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 * min(
                step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]


def copy_config_file(config_file, path):
    config_name = os.path.basename(config_file)
    out_path = os.path.join(path, config_name)
    shutil.copyfile(config_file, out_path)


def load_config(config_path):
    config = AttrDict()
    with open(config_path, "r") as f:
        input_str = f.read()
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    config.update(data)
    return config


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    

def save_checkpoint(model, optimizer, model_loss, out_path,
                    current_step, epoch):
    checkpoint_path = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print(" | | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = model.state_dict()
    state = {
        'model': new_state_dict,
        'optimizer': optimizer.state_dict(),
        'step': current_step,
        'epoch': epoch,
        'loss': model_loss,
        'date': datetime.date.today().strftime("%B %d, %Y")
    }
    torch.save(state, checkpoint_path)


def check_update(model, grad_clip):
    r'''Check model gradient against unexpected jumps and failures'''
    skip_flag = False
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if np.isinf(grad_norm):
        print(" | > Gradient is INF !!")
        skip_flag = True
    return grad_norm, skip_flag
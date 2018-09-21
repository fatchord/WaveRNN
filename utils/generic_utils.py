import os
import json
import datetime
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    config.update(json.load(open(config_path, "r")))
    return config


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
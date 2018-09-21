import os
import json
import datetime
import torch


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AnnealLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super(AnnealLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr * self.warmup_steps**0.5 * min(
                step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]

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
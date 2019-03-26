import time
from torch import optim
import torch.nn.functional as F
from utils.display import stream, simple_table
from utils.dataset import get_datasets
import hparams as hp
from models.fatchord_wavernn import Model
from generate import gen_testset
from utils.paths import Paths
import argparse

parser = argparse.ArgumentParser(description='Train WaveRNN')
parser.add_argument('--lr', '-l', type=float, default=hp.lr, help='[float] override hparams.py learning rate')
parser.add_argument('--batch_size', '-b', type=int, default=hp.batch_size, help='[int] override hparams.py batch size')
parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train regardless of total_steps')
args = parser.parse_args()

batch_size = args.batch_size
force_train = args.force_train
lr = args.lr


def train_loop(model, optimiser, train_set, test_set, lr, total_steps):

    for p in optimiser.param_groups: p['lr'] = lr

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1

    for e in range(1, epochs + 1):

        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.cuda(), m.cuda(), y.cuda()

            y_hat = model(x, m)
            y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            y = y.unsqueeze(-1)
            loss = F.cross_entropy(y_hat, y)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            speed = i / (time.time() - start)
            avg_loss = running_loss / i

            step = model.get_step()
            k = step // 1000

            if step % hp.checkpoint_every == 0 :
                gen_testset(model, test_set, hp.test_samples, hp.batched, hp.target, hp.overlap, paths.output)
                model.checkpoint(paths.checkpoints)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        model.save(paths.latest_weights)
        model.log(paths.log, msg)
        print(' ')


print('\nInitialising Model...\n')

model = Model(rnn_dims=hp.rnn_dims,
              fc_dims=hp.fc_dims,
              bits=hp.bits,
              pad=hp.pad,
              upsample_factors=hp.upsample_factors,
              feat_dims=hp.num_mels,
              compute_dims=hp.compute_dims,
              res_out_dims=hp.res_out_dims,
              res_blocks=hp.res_blocks,
              hop_length=hp.hop_length,
              sample_rate=hp.sample_rate).cuda()

paths = Paths(hp.data_path, hp.model_id)

model.restore(paths.latest_weights)

optimiser = optim.Adam(model.parameters())

train_set, test_set = get_datasets(paths.data, batch_size)

total_steps = 10_000_000 if force_train else hp.total_steps

simple_table([('Steps Left', str((total_steps - model.get_step())//1000) + 'k'),
              ('Batch Size', batch_size),
              ('LR', lr),
              ('Sequence Len', hp.seq_len))])

train_loop(model, optimiser, train_set, test_set, lr, total_steps)

print('Training Complete. To continue training increase total_steps in hparams.py or use --force_train')
